#!/bin/bash

git add .
git status
echo -e "Enter commit message:"
read MESSAGE
git commit -m "$MESSAGE"