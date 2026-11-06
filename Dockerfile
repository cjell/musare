# munity on Fly.io: the app and nothing it does not serve with.
#
# Data is not in the image. Accounts, histories and the map's artefacts live on
# a Fly volume mounted at /app/data, which survives deploys; the image is only
# code. See scripts/pack_data.py for putting the shared artefacts there.

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Dependencies first, read straight from pyproject, so a code change does not
# reinstall them. Only the default set: the ml extra (umap, scikit-learn) is for
# rebuilding the map offline, and nothing the server runs imports it.
COPY pyproject.toml ./
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))" > /tmp/requirements.txt \
 && pip install -r /tmp/requirements.txt \
 && rm /tmp/requirements.txt

# Not installed as a package: the code finds its data relative to the checkout
# (config.ROOT), so it runs from /app/src with /app/data beside it.
COPY src ./src
COPY profile.html ./

EXPOSE 8080
# Fly's proxy terminates TLS; trusting its forwarded headers keeps request URLs
# and the Secure session cookie right.
CMD ["uvicorn", "musicshare.api.app:app", "--host", "0.0.0.0", "--port", "8080", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
