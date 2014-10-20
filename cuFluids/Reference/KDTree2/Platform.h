#pragma once
#ifndef _KD_PLATFORM_H
#define _KD_PLATFORM_H
/*-----------------------------------------------------------------------------
  Name:	Platform.h
  Desc:	Used to express platform differences
        For Instance:  Windows, UNIX, etc.
  Log:	Created by Shawn D. Brown (3/18/10)
-----------------------------------------------------------------------------*/

/*-------------------------------------
  Platform Definitions
-------------------------------------*/

//
// OS Platform
//
#define KD_PLATFORM_UNIX       1
#define KD_PLATFORM_WIN_VS2008 2

//#define KD_PLATFORM KD_PLATFORM_UNIX
#define KD_PLATFORM KD_PLATFORM_WIN_VS2008


//
// CPU Platform (Intel or ???)
//
#define CPU_UNKNOWN 0
#define CPU_INTEL_X86     1
#define CPU_INTEL_X64     2

//#define CPU_PLATFORM CPU_UNKNOWN
//#define CPU_PLATFORM CPU_INTEL_X86
#define CPU_PLATFORM CPU_INTEL_X64

#endif // _KD_PLATFORM_H

