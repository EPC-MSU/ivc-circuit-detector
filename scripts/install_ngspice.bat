@echo off
setlocal

pushd "%~dp0.."

set "SEVENZIP=C:\Program Files\7-Zip\7z.exe"
set "NGSPICE_ARCHIVE=%CD%\Spice64_dll.7z"
set "NGSPICE_TARGET=%CD%\venv\Lib\site-packages\PySpice\Spice\NgSpice"
set "SYMLINK_DIR=%NGSPICE_TARGET%\Spice64_dll\dll-vs"
set "NETLIST_PY=%CD%\venv\Lib\site-packages\PySpice\Spice\Netlist.py"

echo Extracting NgSpice DLL...
powershell -Command "& '%SEVENZIP%' x '%NGSPICE_ARCHIVE%' '-o%NGSPICE_TARGET%' -y; if ($LASTEXITCODE -ne 0) { exit 1 }"
if errorlevel 1 (
    echo ERROR: Failed to extract. Make sure Spice64_dll.7z is in the repo root and 7-Zip is installed at C:\Program Files\7-Zip.
    goto :done_error
)

echo Creating ngspice.dll symlink...
if exist "%SYMLINK_DIR%\ngspice.dll" (
    echo Symlink already exists, skipping.
) else (
    powershell -Command "Set-Location '%SYMLINK_DIR%'; cmd /c 'mklink ngspice.dll ngspice-34.dll'; if ($LASTEXITCODE -ne 0) { exit 1 }"
    if errorlevel 1 (
        echo ERROR: Failed to create symlink. Make sure you are running as Administrator.
        goto :done_error
    )
)

echo Applying PySpice Netlist.py bug fix...
powershell -Command "$f = '%NETLIST_PY%'; (Get-Content $f -Raw) -replace 'return self\.__class__\(self\._name, self\._model_type, self\._parameters\)', 'return self.__class__(self._name, self._model_type, **self._parameters)' | Set-Content $f"
if errorlevel 1 (
    echo ERROR: Failed to patch Netlist.py.
    goto :done_error
)

echo NgSpice installation complete.
popd
endlocal
exit /b 0

:done_error
popd
endlocal
exit /b 1
