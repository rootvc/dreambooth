#!/bin/bash
set -x

DIR=$(dirname "$0")

launchctl unload ~/Library/LaunchAgents/vc.root.*.plist

cp $DIR/vc.root.*.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/vc.root.*.plist
