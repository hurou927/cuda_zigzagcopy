#ifndef _GETTIME
#define _GETTIME

#define TIME() get_time_sec()

#if __cplusplus >= 201103L  //-std=c++11
	#define __GETTIMEMETHOD "chrono"
	#include <chrono>
	using namespace std::chrono;
	inline double get_time_sec(void){
		return static_cast<double>(duration_cast<nanoseconds>(
			steady_clock::now().time_since_epoch()).count())/1000000000;
	}

#elif defined(__unix)
	#define __GETTIMEMETHOD "gettimeofday"
	#include <sys/time.h>
	inline double get_time_sec(void){
    	struct timeval tv;
   		gettimeofday(&tv,NULL);
    	return((double)(tv.tv_sec)+(double)(tv.tv_usec)/1000000);
	}

#elif defined(_WIN32)
	#define __GETTIMEMETHOD "QueryPerformanceCounter"
	#include <windows.h>
	inline double get_time_sec(void){
    	LARGE_INTEGER freq,time;
    	QueryPerformanceFrequency(&freq);
    	QueryPerformanceCounter(&time);
    	return((double)time.QuadPart/freq.QuadPart);
	}
#endif

#ifdef __cplusplus
	#define __DEFAULT_TIME_UNIT "s"
	#define __DEFAULT_MAX_EVENT_NUM 20


	#include <vector>
	#include <iostream>
	#include <cstdio>
	#include <string>
	#include <map>
	using namespace std;


	class timeStamp{
	public:
		timeStamp()                           { initialize(__DEFAULT_MAX_EVENT_NUM ,__DEFAULT_TIME_UNIT); };
		timeStamp(int num)                    { initialize(num                     ,__DEFAULT_TIME_UNIT); };
		timeStamp(string time_unit)           { initialize(__DEFAULT_MAX_EVENT_NUM ,time_unit); };
		timeStamp(int num, string time_unit)  { initialize(num                     ,time_unit); };
		double& operator[](int i)             { return sec_vec[i];} ;
		void operator()()                     { sec_vec[timeCount++]=get_time_sec();};
		void stamp()                          { sec_vec[timeCount++]=get_time_sec();};
		double interval(int index1,int index2){ return (sec_vec[index2]-sec_vec[index1]); };
		double  sum()                         { return sec_vec[timeCount-1]-sec_vec[0]; };
		/*int     gettimeCount()                { return timeCount; };
    	string  getunit()                     { return unit; };
    	double  getxrate()                    { return xrate; } ;
		int     getlimit()                    { return limitOfTimeCount; } ;*/

		friend ostream& operator<<(ostream& os, timeStamp &ts);
		void print() {cout<<*this;};
	private:
		void initialize(int limitOfTimeCount,std::string time_unit);
		//double xrateFromUnit();
		vector <double> sec_vec;
		int limitOfTimeCount;
		int timeCount;
	    string unit;//"ms" or "s"(default)
	    double xrate;
	};
	void timeStamp::initialize(int i,std::string time_unit){

		sec_vec.resize(i);
		limitOfTimeCount=i;
		timeCount=0;
		unit=time_unit;
		if     (unit=="ns")    xrate=1000000000;
		else if(unit=="micro") xrate=1000000;
		else if(unit=="ms")    xrate=1000;
		else if(unit=="m")     xrate=(1.0/60);
		else if(unit=="h")     xrate=(1.0/3600);
		else                  {xrate=1;unit="s";}
	}

	ostream& operator<<(ostream& os, timeStamp &ts){
		for(int i=1;i<ts.timeCount;i++)
			cout<<fixed <<(ts[i]-ts[i-1])*ts.xrate<<" , "<<ts.unit<<endl;
		return os;
	}

#endif


#endif

