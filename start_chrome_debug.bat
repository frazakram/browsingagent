@echo off
echo Starting Chrome with Remote Debugging on port 9222...
echo.
echo After Chrome opens, you can use it normally.
echo The browsing agent will connect to this Chrome instance.
echo.

:: Try common Chrome installation paths
if exist "C:\Program Files\Google\Chrome\Application\chrome.exe" (
    start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222
    goto :done
)

if exist "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" (
    start "" "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222
    goto :done
)

:: Try using the default 'chrome' command
start chrome --remote-debugging-port=9222
goto :done

:done
echo Chrome started with remote debugging enabled!
echo You can now run your browsing agent.
pause

