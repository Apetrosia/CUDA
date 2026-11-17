@echo off
chcp 65001 >nul
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvarsall.bat" x64
nvcc TEST.cu -o TEST.exe
if %errorlevel% equ 0 (
    echo.
    echo Компиляция успешна! Запускаем программу...
    echo.
    TEST.exe
) else (
    echo.
    echo Ошибка компиляции!
)
pause