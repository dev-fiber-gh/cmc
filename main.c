#include <time.h> 
#include <stdio.h>

#include <stdlib.h>
#include <string.h> 


#define MakeContexts \
        const unsigned char prev8 = (unsigned char)History;\
        const unsigned h8 = history8[prev8];\
        const unsigned short prev16 = (unsigned short)History;\
        const unsigned h16 = history16[prev16];\
        const unsigned prev24 = 0xffffff & History;\
        const unsigned h24 = history24[prev24];\
        unsigned FNV1a = 0x811c9dc5;\
        FNV1a ^= History & 0xff;\
        FNV1a *= 0x1000193;\
        FNV1a ^= History >> 8 & 0xff;\
        FNV1a *= 0x1000193;\
        FNV1a ^= History >> 16 & 0xff;\
        FNV1a *= 0x1000193;\
        FNV1a ^= History >> 24 & 0xff;\
        FNV1a *= 0x1000193;\
        const unsigned prev32 = 0x3ffffff & FNV1a;\
        const unsigned h32 = history32[prev32];\
        FNV1a ^= History >> 32 & 0xff;\
        FNV1a *= 0x1000193;\
        const unsigned prev40 = 0x3ffffff & FNV1a;\
        const unsigned h40 = history40[prev40];\
        FNV1a ^= History >> 40 & 0xff;\
        FNV1a *= 0x1000193;\
        const unsigned prev48 = 0x3ffffff & FNV1a;\
        const unsigned h48 = history48[prev48];\
        FNV1a ^= History >> 48 & 0xff;\
        FNV1a *= 0x1000193;\
        const unsigned prev56 = 0x3ffffff & FNV1a;\
        const unsigned h56 = history56[prev56];\

#define SetX\
        x[ 0] = lg[GetProbability(Model1, (unsigned char)History)];\
        x[ 1] = lg[GetProbability(Model1, (unsigned char)h8)];\
        x[ 2] = lg[GetProbability(Model1, (unsigned char)h16)];\
        x[ 3] = lg[GetProbability(Model1, (unsigned char)h24)];\
        x[ 4] = lg[GetProbability(Model1, (unsigned char)h32)];\
        x[ 5] = lg[GetProbability(Model1, (unsigned char)h40)];\
        x[ 6] = lg[GetProbability(Model1, (unsigned char)h48)];\
        x[ 7] = lg[GetProbability(Model1, (unsigned char)h56)];\
        x[ 8] = lg[GetProbability(Model2, (unsigned short)History)];\
        x[ 9] = lg[GetProbability(Model2, (unsigned short)h8)];\
        x[10] = lg[GetProbability(Model2, (unsigned short)h16)];\
        x[11] = lg[GetProbability(Model2, (unsigned short)h24)];\
        x[12] = lg[GetProbability(Model2, (unsigned short)h32)];\
        x[13] = lg[GetProbability(Model2, (unsigned short)h40)];\
        x[14] = lg[GetProbability(Model2, (unsigned short)h48)];\
        x[15] = lg[GetProbability(Model2, (unsigned short)h56)];\
        x[16] = lg[GetProbability(Model3, 0xffffff & History)];\
        x[17] = lg[GetProbability(Model3, 0xffffff & h8)];\
        x[18] = lg[GetProbability(Model3, 0xffffff & h16)];\
        x[19] = lg[GetProbability(Model3, 0xffffff & h24)];\
        x[20] = lg[GetProbability(Model3, 0xffffff & h32)];\
        x[21] = lg[GetProbability(Model3, 0xffffff & h40)];\
        x[22] = lg[GetProbability(Model3, 0xffffff & h48)];\
        x[23] = lg[GetProbability(Model3, 0xffffff & h56)];
        
#define UpdateModels\
        UpdateModel(Model1, (unsigned char)History, y);\
        UpdateModel(Model1, (unsigned char)h8,  y);\
        UpdateModel(Model1, (unsigned char)h16, y);\
        UpdateModel(Model1, (unsigned char)h24, y);\
        UpdateModel(Model1, (unsigned char)h32, y);\
        UpdateModel(Model1, (unsigned char)h40, y);\
        UpdateModel(Model1, (unsigned char)h48, y);\
        UpdateModel(Model1, (unsigned char)h56, y);\
        UpdateModel(Model2, (unsigned short)History, y);\
        UpdateModel(Model2, (unsigned short)h8,  y);\
        UpdateModel(Model2, (unsigned short)h16, y);\
        UpdateModel(Model2, (unsigned short)h24, y);\
        UpdateModel(Model2, (unsigned short)h32, y);\
        UpdateModel(Model2, (unsigned short)h40, y);\
        UpdateModel(Model2, (unsigned short)h48, y);\
        UpdateModel(Model2, (unsigned short)h56, y);\
        UpdateModel(Model3, 0xffffff & History, y);\
        UpdateModel(Model3, 0xffffff & h8,  y);\
        UpdateModel(Model3, 0xffffff & h16, y);\
        UpdateModel(Model3, 0xffffff & h24, y);\
        UpdateModel(Model3, 0xffffff & h32, y);\
        UpdateModel(Model3, 0xffffff & h40, y);\
        UpdateModel(Model3, 0xffffff & h48, y);\
        UpdateModel(Model3, 0xffffff & h56, y);\
        history8[prev8]   = (history8[prev8]   << 1) | y;\
        history16[prev16] = (history16[prev16] << 1) | y;\
        history24[prev24] = (history24[prev24] << 1) | y;\
        history32[prev32] = (history32[prev32] << 1) | y;\
        history40[prev40] = (history40[prev40] << 1) | y;\
        history48[prev48] = (history48[prev48] << 1) | y;\
        history56[prev56] = (history56[prev56] << 1) | y;\
        History = (History << 1) | y;
        
        
//Weight adjustment Wi = Wi + 0.015 * error * Xi
//error = (desired_bit - actual_bit), actual_bit read as probability of a bit 
#define UpdateWeights\
        const float e = 0.015 * ((float)((y << ProbWidth) - p) * multiplier);\
        for (int i = 0; i < NumberOfModels; i++) Weight[i] = Weight[i] + e * x[i];

#define NumberOfModels 24
        
        

#define TooFewArgumentsError 1
#define SameNameArgsError 2
#define InputFileError 3
#define OutputFileError 4
#define NameCollisionError 5
#define DiskError 6


enum Prob { ProbWidth = 15, ProbLearner = 6, ProbMax = (1 << ProbWidth), ProbMod = ProbMax - 1};

#include "log.h"
#include "e.h"

float Sigma;
float x[NumberOfModels];
float Weight[NumberOfModels];
const float multiplier = 1. / ProbMax;

unsigned long long History = 0;
unsigned short Model1[256];
unsigned short Model2[65536];
unsigned short Model3[16777216];
unsigned history8[256];
unsigned history16[65536];
unsigned history24[16777216];
unsigned history32[0x4000000];
unsigned history40[0x4000000];
unsigned history48[0x4000000];
unsigned history56[0x4000000];

unsigned Low = 0;
unsigned Curr = 0;
unsigned High = 0xffffffff;
long long FileSize;


FILE* in, * out; 


void Startup() {
    for (int i = 0; i < 256; i++)      { history8[i]  = 85;      Model1[i] = ProbMax / 2; }
    for (int i = 0; i < 65536; i++)    { history16[i] = 21845;   Model2[i] = ProbMax / 2; }
    for (int i = 0; i < 16777216; i++) { history24[i] = 5592405; Model3[i] = ProbMax / 2; }
    for (int i = 0; i < 0x4000000; i++) history32[i] = history40[i] = history48[i] = history56[i] = 1431655765;
    for (int i = 0; i < NumberOfModels; i++) Weight[i] = 0.5; 
}

void EncodeBit(const int y, const int p) {
    const unsigned mid = Low + (unsigned)((unsigned long long)(High - Low)*((unsigned)p) >> ProbWidth);

    if (y) High = mid;
    else Low = mid + 1;

    while ((High^Low) < 0x1000000)	{
    putc(High >> 24, out);
        High = High << 8 | 255;
        Low = Low << 8;
    }
}
char DecodeBit(const int p) {
    int c;
    char y;
    const unsigned mid = Low + (unsigned)(((unsigned long long)(High - Low)*(unsigned)p) >> ProbWidth);

    Curr <= mid ? (y = 1, High = mid) : (y = 0, Low = mid + 1);

    while ((High^Low) < 0x1000000) {
        High = High << 8 | 255;
        Low = Low << 8;
        c = getc(in);
        Curr = Curr << 8 | c;
    }
    return y;
}


int CalculateMixedPrediction() {
    Sigma = 0;
    for (int i = 0; i < NumberOfModels; i++) Sigma += x[i] * Weight[i];
    
    int result;
    if(Sigma >= 0) {
        if (Sigma > 119) Sigma = 119;
        result = SPositive[(unsigned char)Sigma];
    }
    else {
        if (Sigma < -119) Sigma = -119;
        result = SNegative[(unsigned char)-Sigma];
    }
 
    return result;
}
int GetProbability(unsigned short model[], const unsigned context) {
    return model[context] >> ProbWidth ? model[context] = ProbMod : model[context];
}
void UpdateModel(unsigned short model[], const unsigned context, const int y) {
    model[context] += ((y << ProbWidth) - model[context]) >> ProbLearner;
}


void CompressChar(const int c) {
    int p;
    for (int i = 7; i >= 0; --i) {
        const int y = c >> i & 1;
        MakeContexts        
        SetX
        UpdateModels
        p = CalculateMixedPrediction();
        UpdateWeights
        EncodeBit(y, p);
    }
}
int DecompressChar() {
    int p;
    int y;
    int c = 0;
    for (int i = 7; i >= 0; i--) {
        MakeContexts
        SetX
        p = CalculateMixedPrediction();
        y = DecodeBit(p);
        UpdateWeights
        UpdateModels
        c += y << i;
    }
    return c;
}



long long GetFileSize(FILE* _in) {
    fseek(_in, 0, SEEK_END);
    const long long len = ftell(_in);
    rewind(_in);
    return len;
}


int main(int argc, char* argv[]) {
    clock_t Start = clock();

    if (argc < 4) {
        printf("Usage example:\n");
        printf("\t%s c infile outfile\n", argv[0]);
        printf("\t%s d infile outfile\n", argv[0]);
        exit(TooFewArgumentsError);
    }
    
    
    if (!strcmp(argv[2], argv[3])) { 
        printf("Argument error. The first and second file names cannot be the same.\n"); exit(SameNameArgsError); 
    }

    if (!strcmp("c", argv[1])) {
        if (NULL == (in = fopen(argv[2], "rb")))  {
            printf("Argument error. The specified file could not be found (%s).\n", argv[2]); exit(InputFileError);
        }
        if((fopen(argv[3], "rb"))) { 
            printf("Name collision. The output file name matches an existing file.\n"); exit(NameCollisionError);
        }
        if (NULL == (out = fopen(argv[3], "wb"))) {
            printf("Disk error. Unable to open the file (%s).\n", argv[3]); exit(DiskError);
        }

        FileSize = GetFileSize(in);        
        fwrite(&FileSize, 8, 1, out);
        
        Startup();       
        
        int c; long long i = 0;
        while(EOF != (c = getc(in))) {
            CompressChar(c); 
            if(0 == (i & 0xffff)) printf("%lld\n", i);
            i++;
        }

        putc(High >> 24, out);
        putc(High >> 16, out);
        putc(High >> 8, out);
        putc(0xff & High, out);

        fclose(in);
        fclose(out);
    }
    else if (!strcmp("d", argv[1])) {
        if (NULL == (in = fopen(argv[2], "rb")))  {
            printf("Argument error. The specified file could not be found (%s).\n", argv[2]);  exit(InputFileError);
        }
        if(fopen(argv[3], "rb")) { 
            printf("Name collision. The output file name matches an existing file.\n"); exit(NameCollisionError);
        }
        if (NULL == (out = fopen(argv[3], "wb"))) {
            printf("Disk error. Unable to open the file (%s).\n", argv[3]); exit(DiskError);
        }

        fread(&FileSize, 8, 1, in);

        Startup();
        
        for (int i = 0; i < 4; ++i) Curr = Curr << 8 | getc(in); 
        for(long long i = 0; i < FileSize; i++) {
            putc(DecompressChar(), out);
            if(0 == (i & 0xffff)) printf("%lld\n", i);
        }
        
        fclose(in);
        fclose(out);
    }

    printf("%f seconds\n", (double)(clock() - Start) / CLOCKS_PER_SEC);

    return 0;
}

