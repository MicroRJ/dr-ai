@REM # For debug builds I can't compile with /Za because windows complains about it...
@SET myCompilerOptions=/options:strict /nologo /TP /Od /MDd /Z7 /WX /W4 /Fa /std:c++20
@SET myLinkerOptions=/DEBUG /SUBSYSTEM:CONSOLE /INCREMENTAL:NO
@SET myInclude=/I. /I.. /I..\vendor\stb

@CALL cl %myCompilerOptions% %myInclude% num-ai.c /link %myLinkerOptions%
