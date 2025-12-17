#!/bin/bash
set -e

# Use passed UID/GID or default to 1000
USER_ID=${HOST_UID:-1000}
GROUP_ID=${HOST_GID:-1000}
USER_NAME=hostuser

# Create group if it doesn't exist
if ! getent group $GROUP_ID >/dev/null; then
    groupadd -g $GROUP_ID $USER_NAME
fi
# Create user if it doesn't exist
if ! id -u $USER_ID >/dev/null 2>&1; then
    useradd -m -u $USER_ID -g $GROUP_ID $USER_NAME
fi

# Change ownership of /models to the user
chown -R $USER_ID:$GROUP_ID /models

# Run the download script as the created user, passing all arguments
exec gosu $USER_ID:$GROUP_ID bash download_with_cli.sh "$@"
