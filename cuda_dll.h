#include <map>
using std::map;
#include <string>
using std::string;

#ifdef CUDADLL_EXPORTS
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT __declspec(dllimport)
#endif

extern "C" DLLEXPORT bool simpleNN(map<string, string> inputMap);