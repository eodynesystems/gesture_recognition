ECHO ON
rem Run a Python script in a given conda environment from a batch file.

rem Define here the path to your conda installation
set CONDAPATH=C:\Users\USUARIO\anaconda3
rem Define here the name of the environment
set ENVNAME=GR_env

rem The following command activates the base environment.
rem call C:\ProgramData\Miniconda3\Scripts\activate.bat C:\ProgramData\Miniconda3
if %ENVNAME%==base (set ENVPATH=%CONDAPATH%) else (set ENVPATH=%CONDAPATH%\envs\%ENVNAME%)

rem Activate the conda environment
rem Using call is required here, see: https://stackoverflow.com/questions/24678144/conda-environments-and-bat-files
call %CONDAPATH%\Scripts\activate.bat %ENVPATH%

python data/collect_data_GUI.py
PAUSE

