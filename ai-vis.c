#define KLARK_APPMODE_WINDOWS
#include "detroit\detroit-app.c"

#include "dr-vec.h"
#include "num-ai.h"

VOID MAIN()
{
  // TODO(RJ): this is temporary, handle incompetence
  SetCurrentDirectoryA("..\\detroit");
  DETROIT_APP App = {};
  App.Modules.App.Name = "dr-ai-vis";
  App.Modules.App.LiveReload.FileSystem = ".";
  App.Modules.App.LiveReload.IsStatic   = TRUE;

  DetroitApp_Init(& App, L"dr-ai-vis");
  SetCurrentDirectoryA("..\\dr-ai");


  unsigned int labels_file_size;
  void *labels_file_data = LoadFileData(& labels_file_size, "data\\train-labels.idx1-ubyte");
  unsigned int label_count = int_flip(((unsigned int *)labels_file_data)[1]);
  unsigned char *label_array = (unsigned char *)labels_file_data + sizeof(int)*2;
  unsigned int images_file_size;
  void *images_file_data = LoadFileData(& images_file_size, "data\\train-images.idx3-ubyte");
  unsigned int img_num = int_flip(((unsigned int *) images_file_data)[1]);
  unsigned int img_row_len = int_flip(((unsigned int *) images_file_data)[2]);
  unsigned int img_col_len = int_flip(((unsigned int *) images_file_data)[3]);
  unsigned char * img_mem = (unsigned char *)images_file_data + sizeof(int)*3;

  int cur_img_idx = 0;



  f32 TimerAtSecond = 0.f;
  while(! App.State.Quit)
  {
    f32 ViewSizeX   = (f32) App.Window.Main.WindowSizeX;
    f32 ViewSizeY   = (f32) App.Window.Main.WindowSizeY;
    f32 AspectRatio = ViewSizeY / ViewSizeX;

    TimerAtSecond += App.Time.Delta.Seconds;

    if(TimerAtSecond >= .255f)
    { cur_img_idx = ((++ cur_img_idx) % img_num);
      TimerAtSecond = 0;
    }

    unsigned char *img_dat = img_mem + (img_col_len * img_row_len) * cur_img_idx;
    unsigned int label = label_array[cur_img_idx];


    f32 Speed = 2.f * App.Time.Delta.Seconds;
    DetroitApp_SetCameraMode(App, DET_CAMERA_FPS);
    DetroitApp_SetCameraSpeed(App, Speed, Speed, Speed);

    DetroitApp_PickMatrix(App, MATRIX_PROJECTION);
    DetroitApp_LoadMatrix(App);
    DetroitApp_MultMatrix(App, MatrixP(AspectRatio, 90.f, 0.001f, 1.f*1000.f));

    DetroitApp_PickMatrix(App, MATRIX_MODELVIEW);

    DetroitApp_LoadMatrix(App);
    DetroitApp_Scale(App, 2.f, 2.f, 1.f);
    DetroitApp_Translate(App, 0.f, 0.f, 2.f);
    DetroitApp_DrawImageData(App, PIXEL_FORMAT_RGB8,img_col_len,img_row_len,img_dat);


    DetroitApp_LoadMatrix(App);
    DetroitApp_Scale(App, 2.f / ViewSizeX, 2.f / ViewSizeY, 1.f);
    DetroitApp_Translate(App, -1.0f, -0.0f, 2.f);

    DetroitApp_DrawText(& App, 64.f, "label: %i", label);

    DetroitApp_Tick(& App);
  }
}