#!/bin/bash
set -e

# Wait for database to be ready
if [ "$DATABASE_URL" ]; then
    echo "Waiting for database..."
    python -c "
import time
import psycopg2
from urllib.parse import urlparse
url = urlparse('$DATABASE_URL')
while True:
    try:
        conn = psycopg2.connect(
            host=url.hostname,
            port=url.port,
            user=url.username,
            password=url.password,
            database=url.path[1:]
        )
        conn.close()
        break
    except psycopg2.OperationalError:
        time.sleep(1)
"
    echo "Database is ready!"
fi

# Run database migrations
echo "Running database migrations..."
python scripts/migrate-db.py

# Create necessary directories
mkdir -p /app/data /app/outputs /app/logs /app/models

# Execute the main command
exec "$@"