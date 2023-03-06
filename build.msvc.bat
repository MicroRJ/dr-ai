@REM # For debug builds I can't compile with /Za because windows complains about it...
@SET myDebugOptions=/Od /MDd
@SET myReleaseOptions=/O2
@SET myCompilerOptions=/options:strict /nologo /TC /Z7 /WX /W4 /Fa -D_DEBUG -D_HARD_DEBUG -D_DEVELOPER
@SET myLinkerOptions=/INCREMENTAL:NO
@CALL taskkill /f /im main.exe
@REM @CALL cl %myCompilerOptions% -D_LANE_256 %myDebugOptions% /TP main.c /link /SUBSYSTEM:CONSOLE %myLinkerOptions%
@CALL cl %myCompilerOptions% -D_LANE_256 %myReleaseOptions% main.c /link /SUBSYSTEM:CONSOLE %myLinkerOptions%
