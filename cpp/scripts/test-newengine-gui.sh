#!/bin/bash

rm $(dirname $0)/../../../Mzinga.LinuxX64/mzingacpp

cp $(dirname $0)/../build/Release/mzingacpp $(dirname $0)/../../../Mzinga.LinuxX64/mzingacpp

./$(dirname $0)/../../../Mzinga.LinuxX64/MzingaViewer

