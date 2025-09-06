#!/bin/bash

# Define the log file
LOG_FILE=$1  #"/var/log/memory_monitor.log"

while true; do
# Get the current date and time
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

# Get MemAvailable from /proc/meminfo
AVAILABLE_MEMORY=$(grep "MemAvailable:" /proc/meminfo | awk '{print $2}')

# Log the information

echo "$TIMESTAMP - Available Memory: $AVAILABLE_MEMORY kB" >> "$LOG_FILE"
sleep 60
done
