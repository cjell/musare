-- Listening capture, running inside Supabase.
--
-- Spotify's recently-played list turned out to omit most of this listener's
-- plays - 15 captured on a day with hours of listening, a song played ten times
-- reported twice - while currently-playing was right every time it was asked.
-- So the history is built here from currently-playing instead: a job asks what
-- is playing every 10 seconds, and a play is written when the track changes,
-- restarts, or stops.
--
-- It asked every 30 seconds until Spotify's account export gave something to
-- measure against: over the nine days both covered, the watcher had missed 19%
-- of plays on its best day and 82% on its first. A song started and skipped
-- between two checks is never seen at all, and short plays are most of what
-- gets skipped. Ten seconds narrows that window without pretending to close
-- it - a track dropped after three seconds is still invisible, and the export
-- stays the way to correct the record.
--
-- It runs in the database rather than on a laptop because the laptop is not
-- online when its owner is walking around with headphones in.
--
--   listen.watch_step    pure: previous state + one snapshot -> next state, finished play
--   listen.watch_poll    every 10s: ask Spotify, step, write
--   listen.recent_rows   pure: a recently-played response -> rows
--   listen.poll_recent   every 30m: the backup, for anything the watcher missed
--   listen.add_play      the one place rows are written, and where the two sources dedupe
--
-- Applied by `scripts/capture.py install`. Safe to apply repeatedly.

create extension if not exists http with schema extensions;
create extension if not exists pg_cron;

create schema if not exists listen;
-- Not exposed through the Supabase API and not reachable by its roles. Only the
-- database's own jobs and a direct connection with the owner's password read it.
revoke all on schema listen from public, anon, authenticated;

create table if not exists listen.plays (
  -- When the play ENDED, which is the convention the export and recently-played
  -- both use, so all three sources sort and dedupe on the same instant.
  played_at   timestamptz not null,
  track_uri   text        not null,
  track_name  text,
  artist_name text,
  album_name  text,
  duration_ms integer,
  ms_played   integer     not null,
  source      text        not null check (source in ('watch', 'recent')),
  recorded_at timestamptz not null default now(),
  primary key (source, track_uri, played_at)
);
create index if not exists plays_played_at on listen.plays (played_at);

-- One row. What was playing at the last poll, as jsonb because it is only ever
-- read and written whole by watch_step.
create table if not exists listen.watch_state (
  id         boolean primary key default true check (id),
  state      jsonb,
  updated_at timestamptz not null default now()
);
insert into listen.watch_state (id) values (true) on conflict do nothing;

-- The hour-long access token. The refresh token that mints it is in Vault.
create table if not exists listen.token (
  id           boolean primary key default true check (id),
  access_token text,
  expires_at   timestamptz
);
insert into listen.token (id) values (true) on conflict do nothing;

-- Whether each job is running and succeeding. A job that stops writes nothing,
-- which looks exactly like a quiet afternoon unless something records the checks.
create table if not exists listen.status (
  job         text primary key,
  last_run_at timestamptz,
  last_ok_at  timestamptz,
  last_error  text,
  runs        bigint not null default 0,
  added       bigint not null default 0
);


-- ------------------------------------------------------------------ spotify

create or replace function listen.access_token() returns text
language plpgsql security definer set search_path = '' as $$
declare
  cached  record;
  cid     text;
  csecret text;
  refresh text;
  resp    extensions.http_response;
  body    jsonb;
begin
  select access_token, expires_at into cached from listen.token where id;
  if cached.access_token is not null and cached.expires_at > now() + interval '60 seconds' then
    return cached.access_token;
  end if;

  select decrypted_secret into cid     from vault.decrypted_secrets where name = 'spotify_client_id';
  select decrypted_secret into csecret from vault.decrypted_secrets where name = 'spotify_client_secret';
  select decrypted_secret into refresh from vault.decrypted_secrets where name = 'spotify_refresh_token';
  if cid is null or csecret is null or refresh is null then
    raise exception 'spotify secrets are missing from vault - run scripts/capture.py install';
  end if;

  perform extensions.http_set_curlopt('CURLOPT_TIMEOUT_MS', '10000');
  resp := extensions.http(row(
    'POST'::extensions.http_method,
    'https://accounts.spotify.com/api/token',
    -- encode() wraps base64 at 76 characters, and id:secret is 65 bytes, so the
    -- header would carry a newline without the replace.
    array[extensions.http_header('Authorization', 'Basic ' ||
      replace(encode(convert_to(cid || ':' || csecret, 'UTF8'), 'base64'), E'\n', ''))],
    'application/x-www-form-urlencoded',
    'grant_type=refresh_token&refresh_token=' || extensions.urlencode(refresh::varchar)
  )::extensions.http_request);
  if resp.status <> 200 then
    raise exception 'token refresh: HTTP % %', resp.status, left(resp.content, 160);
  end if;

  body := resp.content::jsonb;
  update listen.token
     set access_token = body->>'access_token',
         expires_at   = now() + make_interval(secs => coalesce((body->>'expires_in')::int, 3600))
   where id;
  -- Spotify may rotate the refresh token. Keeping the old one would work until
  -- the day it does not, and then every job fails at once.
  if coalesce(body->>'refresh_token', '') not in ('', refresh) then
    perform vault.update_secret(
      (select id from vault.secrets where name = 'spotify_refresh_token'),
      body->>'refresh_token');
  end if;
  return body->>'access_token';
end $$;


create or replace function listen.spotify_get(url text) returns extensions.http_response
language plpgsql security definer set search_path = '' as $$
declare
  resp extensions.http_response;
begin
  perform extensions.http_set_curlopt('CURLOPT_TIMEOUT_MS', '10000');
  resp := extensions.http(row(
    'GET'::extensions.http_method, url,
    array[extensions.http_header('Authorization', 'Bearer ' || listen.access_token())],
    null, null
  )::extensions.http_request);
  if resp.status = 401 then
    -- Revoked or expired early. Forget it so the next poll refreshes instead of
    -- failing with the same token until the hour is up.
    update listen.token set access_token = null, expires_at = null where id;
  end if;
  return resp;
end $$;


-- ------------------------------------------------------------------- writes

-- Whether two rows are the same recording. The id alone is not enough: Spotify
-- files one song under several ids - the album cut and the single - and the two
-- endpoints do not agree on which to report. Power Trip came back as 7FOJvA3P...
-- from currently-playing and 2uwnP6tZ... from recently-played, 0.2 seconds
-- apart, and was stored twice.
create or replace function listen._same_track(
  a_uri text, a_name text, a_artist text, b_uri text, b_name text, b_artist text
) returns boolean
language sql immutable set search_path = '' as $$
  select a_uri = b_uri
      or (lower(btrim(a_name)) = lower(btrim(b_name))
          and lower(btrim(a_artist)) = lower(btrim(b_artist)))
$$;


-- How close a recently-played row and a watcher row for the same track must be
-- to count as one play. Measured on the first afternoon both ran: the watcher's
-- end time landed within 0.03-0.21 seconds of Spotify's on every play both saw,
-- because it dates the end from the moment the next track started. Thirty
-- seconds is wide of that and still safe for repeats - recently-played never
-- lists a play under about 30 seconds, so two real plays of one song cannot
-- both appear in it this close together.
create or replace function listen.add_play(p jsonb, src text) returns integer
language plpgsql security definer set search_path = '' as $$
declare
  at     timestamptz := (p->>'played_at')::timestamptz;
  uri    text        := p->>'track_uri';
  name   text        := p->>'track_name';
  artist text        := p->>'artist_name';
  n      integer;
begin
  -- The watcher measured this play; the backup only estimated it.
  if src = 'recent' and exists (
    select 1 from listen.plays w
     where w.source = 'watch'
       and w.played_at between at - interval '30 seconds' and at + interval '30 seconds'
       and listen._same_track(w.track_uri, w.track_name, w.artist_name, uri, name, artist)
  ) then
    return 0;
  end if;

  insert into listen.plays
    (played_at, track_uri, track_name, artist_name, album_name, duration_ms, ms_played, source)
  values
    (at, uri, p->>'track_name', p->>'artist_name', p->>'album_name',
     (p->>'duration_ms')::int, (p->>'ms_played')::int, src)
  on conflict do nothing;
  get diagnostics n = row_count;

  if n > 0 and src = 'watch' then
    delete from listen.plays r
     where r.source = 'recent'
       and r.played_at between at - interval '30 seconds' and at + interval '30 seconds'
       and listen._same_track(r.track_uri, r.track_name, r.artist_name, uri, name, artist);
  end if;
  return n;
end $$;


-- ------------------------------------------------------------------ watcher

-- Given what was playing at the last poll and what is playing now, the next
-- state and the play that just finished, if one did.
--
-- Pure - no tables, no clock, no network - so every transition is testable by
-- handing it two snapshots.
--
-- Listening time is accumulated from progress between polls, never from the
-- track's length: a skip at 40 seconds is 40 seconds. Progress is only believed
-- up to the wall time that passed, so seeking forward does not count as having
-- heard the part skipped over.
create or replace function listen.watch_step(s jsonb, snap jsonb, at timestamptz) returns jsonb
language plpgsql stable set search_path = '' as $$
declare
  item      jsonb := snap->'item';
  cur       jsonb;
  prog      bigint;
  playing   boolean;
  elapsed   bigint;
  prev_prog bigint;
  dur       bigint;
  heard     bigint;
  was_on    boolean;
  extra     bigint;
  base      timestamptz;
  play      jsonb;
  -- Spotify can blink out for a poll between tracks or during a device
  -- handoff. Ending a play on the first empty answer and starting it again on
  -- the next would count one listen twice, so nothing ends until it persists.
  -- Five missed checks at the ten-second cadence, and it only delays the end
  -- of a play rather than moving it: the end time comes from when the track
  -- last moved, not from when the blackout was noticed.
  gone_for  constant interval := interval '50 seconds';
begin
  if jsonb_typeof(item) = 'object' and coalesce(item->>'uri', '') like 'spotify:track:%' then
    prog    := greatest(0, coalesce((snap->>'progress_ms')::bigint, 0));
    playing := coalesce((snap->>'is_playing')::boolean, false);
    cur := jsonb_build_object(
      'track_uri',   item->>'uri',
      'track_name',  item->>'name',
      'artist_name', (select string_agg(a->>'name', ', ' order by o)
                        from jsonb_array_elements(item->'artists') with ordinality as x(a, o)),
      'album_name',  item->'album'->>'name',
      'duration_ms', (item->>'duration_ms')::bigint);
  end if;
  -- Episodes, ads and local files have no track uri and read as nothing playing.

  if s is null or s->>'track_uri' is null then
    if cur is null then
      return jsonb_build_object('state', null, 'play', null);
    end if;
    -- First sighting with no history: take what has played so far as heard.
    return jsonb_build_object('state', cur || jsonb_build_object(
      'progress_ms', prog, 'heard_ms', prog, 'playing', playing,
      'seen_at', at, 'moved_at', at), 'play', null);
  end if;

  elapsed   := greatest(0, (extract(epoch from at - (s->>'seen_at')::timestamptz) * 1000)::bigint);
  prev_prog := (s->>'progress_ms')::bigint;
  dur       := coalesce((s->>'duration_ms')::bigint, prev_prog);
  heard     := (s->>'heard_ms')::bigint;
  was_on    := (s->>'playing')::boolean;

  -- ------------------------------------------------ same track still there
  if cur is not null and cur->>'track_uri' = s->>'track_uri' then
    if prog + 3000 < prev_prog and prog <= elapsed + 3000 then
      -- Back near the start, with only enough time gone to have started over:
      -- a repeat, or "previous" pressed to restart. Either way a new play.
      extra := case when was_on then greatest(0, least(dur - prev_prog, elapsed - prog)) else 0 end;
      base  := case when was_on then (s->>'seen_at')::timestamptz else (s->>'moved_at')::timestamptz end;
      play  := listen._finished(s, heard + extra, base + make_interval(secs => extra / 1000.0));
      return jsonb_build_object('state', cur || jsonb_build_object(
        'progress_ms', prog, 'heard_ms', prog, 'playing', playing,
        'seen_at', at, 'moved_at', at), 'play', play);
    end if;

    -- Moving forward counts up to the time that passed; backwards counts nothing.
    extra := case when prog >= prev_prog then least(prog - prev_prog, elapsed + 2000) else 0 end;
    return jsonb_build_object('state', (s - 'gone_at') || jsonb_build_object(
      'progress_ms', prog, 'heard_ms', heard + extra, 'playing', playing, 'seen_at', at,
      'moved_at', case when extra > 0 then to_jsonb(at) else s->'moved_at' end), 'play', null);
  end if;

  -- ------------------------------------------------ nothing playing
  if cur is null then
    if s->>'gone_at' is null then
      return jsonb_build_object('state', s || jsonb_build_object('gone_at', at), 'play', null);
    end if;
    if at - (s->>'gone_at')::timestamptz < gone_for then
      return jsonb_build_object('state', s, 'play', null);
    end if;
    -- It played on until it stopped, somewhere before the first empty poll.
    elapsed := greatest(0, (extract(epoch from (s->>'gone_at')::timestamptz
                                               - (s->>'seen_at')::timestamptz) * 1000)::bigint);
    extra := case when was_on then greatest(0, least(dur - prev_prog, elapsed)) else 0 end;
    base  := case when was_on then (s->>'seen_at')::timestamptz else (s->>'moved_at')::timestamptz end;
    return jsonb_build_object('state', null, 'play',
      listen._finished(s, heard + extra, base + make_interval(secs => extra / 1000.0)));
  end if;

  -- ------------------------------------------------ a different track
  -- The new track started `prog` ago, so the old one ended then - bounded by
  -- how much of it was left.
  extra := case when was_on then greatest(0, least(dur - prev_prog, elapsed - prog)) else 0 end;
  base  := case when was_on then (s->>'seen_at')::timestamptz else (s->>'moved_at')::timestamptz end;
  play  := listen._finished(s, heard + extra, base + make_interval(secs => extra / 1000.0));
  return jsonb_build_object('state', cur || jsonb_build_object(
    -- It can only have played for as long as has passed since the last poll.
    'progress_ms', prog, 'heard_ms', least(prog, elapsed + 2000), 'playing', playing,
    'seen_at', at, 'moved_at', at), 'play', play);
end $$;


create or replace function listen._finished(s jsonb, heard bigint, ended timestamptz) returns jsonb
language sql stable set search_path = '' as $$
  -- Under a second is a track flicking past, not a play.
  select case when least(heard, coalesce((s->>'duration_ms')::bigint, heard)) < 1000 then null
  else jsonb_build_object(
    'played_at',   ended,
    'track_uri',   s->>'track_uri',
    'track_name',  s->>'track_name',
    'artist_name', s->>'artist_name',
    'album_name',  s->>'album_name',
    'duration_ms', (s->>'duration_ms')::bigint,
    'ms_played',   least(heard, coalesce((s->>'duration_ms')::bigint, heard)))
  end
$$;


create or replace function listen.watch_poll() returns void
language plpgsql security definer set search_path = '' as $$
declare
  resp  extensions.http_response;
  snap  jsonb;
  prev  jsonb;
  step  jsonb;
  added integer := 0;
begin
  if not pg_try_advisory_xact_lock(hashtext('listen.watch_poll')) then
    return;  -- the previous poll is still running
  end if;

  begin
    resp := listen.spotify_get('https://api.spotify.com/v1/me/player/currently-playing');
  exception when others then
    perform listen._record('watch', false, sqlerrm, 0);
    return;
  end;

  -- An error is not "nothing playing". Reading it as silence would end the
  -- current play early and count it again when the next poll sees it.
  if resp.status not in (200, 204) then
    perform listen._record('watch', false,
      format('currently-playing: HTTP %s %s', resp.status, left(resp.content, 160)), 0);
    return;
  end if;
  snap := case when resp.status = 200 then nullif(btrim(resp.content), '')::jsonb end;

  select state into prev from listen.watch_state where id for update;
  -- The request took real time; the transaction's now() is from before it.
  step := listen.watch_step(prev, snap, clock_timestamp());
  update listen.watch_state
     set state = nullif(step->'state', 'null'::jsonb), updated_at = now()
   where id;
  if jsonb_typeof(step->'play') = 'object' then
    added := listen.add_play(step->'play', 'watch');
  end if;
  perform listen._record('watch', true, null, added);
end $$;


-- ------------------------------------------------------------------- backup

-- Rows from one recently-played response. It carries no listening time, so it
-- is estimated: played_at marks the end of a play, and what was heard is the
-- smaller of the track's length and the gap since the play before it.
create or replace function listen.recent_rows(body jsonb)
returns table (played_at timestamptz, track_uri text, track_name text, artist_name text,
               album_name text, duration_ms integer, ms_played integer)
language sql stable set search_path = '' as $$
  with items as (
    select (i->>'played_at')::timestamptz as ended, i->'track' as t
      from jsonb_array_elements(coalesce(body->'items', '[]'::jsonb)) as i
     where coalesce(i->'track'->>'uri', '') like 'spotify:track:%'
  ), ordered as (
    select ended, t, lag(ended) over (order by ended) as before from items
  )
  select ended,
         t->>'uri',
         t->>'name',
         (select string_agg(a->>'name', ', ' order by o)
            from jsonb_array_elements(t->'artists') with ordinality as x(a, o)),
         t->'album'->>'name',
         (t->>'duration_ms')::int,
         case when before is null then (t->>'duration_ms')::int
              else greatest(0, least((t->>'duration_ms')::int,
                                     (extract(epoch from ended - before) * 1000)::int)) end
    from ordered
   order by ended
$$;


create or replace function listen.poll_recent() returns void
language plpgsql security definer set search_path = '' as $$
declare
  resp  extensions.http_response;
  r     record;
  added integer := 0;
begin
  if not pg_try_advisory_xact_lock(hashtext('listen.poll_recent')) then
    return;
  end if;
  begin
    resp := listen.spotify_get('https://api.spotify.com/v1/me/player/recently-played?limit=50');
  exception when others then
    perform listen._record('recent', false, sqlerrm, 0);
    return;
  end;
  if resp.status <> 200 then
    perform listen._record('recent', false,
      format('recently-played: HTTP %s %s', resp.status, left(resp.content, 160)), 0);
    return;
  end if;
  for r in select * from listen.recent_rows(resp.content::jsonb) loop
    added := added + listen.add_play(to_jsonb(r), 'recent');
  end loop;
  perform listen._record('recent', true, null, added);
end $$;


create or replace function listen._record(which text, ok boolean, err text, n integer) returns void
language sql security definer set search_path = '' as $$
  insert into listen.status as st (job, last_run_at, last_ok_at, last_error, runs, added)
  values (which, now(), case when ok then now() end, left(err, 300), 1, n)
  on conflict (job) do update set
    last_run_at = excluded.last_run_at,
    last_ok_at  = coalesce(excluded.last_ok_at, st.last_ok_at),
    last_error  = excluded.last_error,
    runs        = st.runs + 1,
    added       = st.added + excluded.added
$$;


revoke all on all tables    in schema listen from public, anon, authenticated;
revoke all on all functions in schema listen from public, anon, authenticated;


-- -------------------------------------------------------------------- jobs

-- cron.schedule replaces a job of the same name, so re-applying this file
-- updates the schedule rather than adding a second copy.
select cron.schedule('listen-watch',  '10 seconds',   'select listen.watch_poll()');
select cron.schedule('listen-recent', '*/30 * * * *', 'select listen.poll_recent()');
-- A job every 10 seconds is 8,640 run records a day in pg_cron's own log.
select cron.schedule('listen-trim-cron-log', '17 4 * * *',
  $$delete from cron.job_run_details where end_time < now() - interval '2 days'$$);


-- --------------------------------------------------------------- keepalive

-- Free projects pause after a week without activity, and a paused project runs
-- no jobs - capture would stop silently. Supabase's docs describe activity as
-- requests from users and applications and do not say whether a project's own
-- cron jobs count, so this does not rely on them: a daily GitHub Actions
-- workflow calls this through the public API, which is unambiguously a request.
--
-- It writes rather than only reading, which covers the stricter reading of
-- "activity", and it records itself in listen.status so `capture.py status`
-- shows when the last one arrived. Callable with the anon key; the most anyone
-- holding that key can do with it is bump one counter.
create or replace function public.keepalive() returns timestamptz
language plpgsql security definer set search_path = '' as $$
begin
  perform listen._record('keepalive', true, null, 0);
  return now();
end $$;
revoke all on function public.keepalive() from public;
grant execute on function public.keepalive() to anon;
notify pgrst, 'reload schema';
