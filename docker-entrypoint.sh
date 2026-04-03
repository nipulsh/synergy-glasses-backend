#!/bin/sh
set -e
mkdir -p /data/dataset /data/models
rm -rf /app/dataset /app/models
ln -s /data/dataset /app/dataset
ln -s /data/models /app/models
exec "$@"
