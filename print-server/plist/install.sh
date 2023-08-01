#!/bin/bash

DIR=$(dirname "$0")

cp $DIR/vc.root.*.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/vc.root.*.plist
