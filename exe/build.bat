REM copy entry script
copy .\entry.py ..\api\entry.py

REM build bundle
pyinstaller win10.directml.dir.spec --noconfirm

REM copy additional dirs
xcopy .\gfpgan .\dist\server\gfpgan /s /e /f /i /y
xcopy ..\api\gui .\dist\client /s /e /f /i /y
xcopy ..\api\schemas .\dist\schemas /s /e /f /i /y
xcopy ..\docs .\dist\docs /s /e /f /i /y
xcopy ..\models .\dist\models /s /e /f /i /y
xcopy ..\outputs .\dist\outputs /s /e /f /i /y

REM copy loose files
copy .\onnx-web-full.bat .\dist\onnx-web-full.bat /y
copy .\onnx-web-half.bat .\dist\onnx-web-half.bat /y
copy .\README.txt .\dist\README.txt
copy ..\api\logging.yaml .\dist\logging.yaml /y
copy ..\api\params.json .\dist\params.json /y

REM set version number
set BUNDLE_TYPE=rc
set BUNDLE_VERSION=0.11.0

REM get commit info
git rev-parse HEAD > commit.txt
set /p GIT_SHA=<commit.txt
set GIT_HEAD=%GIT_SHA:~0,8%

REM create version file
echo "%BUNDLE_VERSION%-%BUNDLE_TYPE%-%GIT_HEAD%" > .\dist\version.txt

REM create archive
"C:\Program Files\7-Zip\7z.exe" a .\dist\onnx-web-v%BUNDLE_VERSION%-%BUNDLE_TYPE%-%GIT_HEAD%.zip ".\dist\*"
