#include "Setting.h"

namespace db {
void Setting::reset() {
    Format = "";
    BookshelfVariety = "";
    BookshelfAux = "";
    BookshelfPl = "";
    DefFile = "";
    LefFile = "";
    LefCell = "";
    LefTech = "";
    Constraints = "";
    Verilog = "";
    OutputFile = "";
    
    liteMode = true;
    random_place = false;
}

Setting setting;

}  //   namespace db
