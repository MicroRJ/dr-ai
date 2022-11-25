@REM # For debug builds I can't compile with /Za because windows complains about it...
@SET myDebugOptions=/Od /MDd
@SET myReleaseOptions=/O2
@SET myCompilerOptions=/options:strict /nologo /TP /Z7 /WX /W4 /Fa /std:c++20
@SET myLinkerOptions=/INCREMENTAL:NO
@SET myInclude=/I. /I.. /I..\vendor\stb

@CALL cl %myCompilerOptions% -DDRL_256 %myDebugOptions% %myInclude% num-ai.c /link /SUBSYSTEM:CONSOLE /DEBUG %myLinkerOptions%
@CALL cl %myCompilerOptions% -DDRL_256 %myReleaseOptions% %myInclude% /arch:AVX num-ai.c /Fenum-ai-avx.exe /link /SUBSYSTEM:CONSOLE %myLinkerOptions%
@REM @CALL cl %myCompilerOptions% %myDebugOptions% %myInclude% /arch:AVX ai-vis.c /link /SUBSYSTEM:WINDOWS %myLinkerOptions%
