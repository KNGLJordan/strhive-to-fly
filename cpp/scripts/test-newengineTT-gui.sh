#!/bin/bash

rm $(dirname $0)/../../../Mzinga.LinuxX64/cppTTZobrist

mv $(dirname $0)/../build/Release/mzingacpp $(dirname $0)/../../../Mzinga.LinuxX64/cppTTZobrist

./$(dirname $0)/../../../Mzinga.LinuxX64/MzingaViewer
