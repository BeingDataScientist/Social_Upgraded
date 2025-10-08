@echo off
echo ========================================
echo  Stopping Digital Media Assessment App
echo ========================================
echo.

echo Stopping Docker containers...
docker-compose down

echo.
echo Application stopped successfully!
echo.
pause
