@echo off
echo Setting up GitHub repository...
echo.
echo Please enter your GitHub username:
set /p GITHUB_USER=

echo.
echo Adding remote origin...
"C:\Program Files\Git\bin\git.exe" remote add origin https://github.com/%GITHUB_USER%/investment-engine.git

echo.
echo Pushing to GitHub...
"C:\Program Files\Git\bin\git.exe" branch -M main
"C:\Program Files\Git\bin\git.exe" push -u origin main

echo.
echo GitHub setup complete!
pause
