# Install script for directory: /mnt/c/Functional_Encryption_for_Logistic_Regression/baselines/gc_ag2pc/emp-tool

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/mnt/c/Functional_Encryption_for_Logistic_Regression/baselines/gc_ag2pc/emp-toolbuild/libemp-tool.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/mnt/c/Functional_Encryption_for_Logistic_Regression/baselines/gc_ag2pc/emp-tool/emp-tool" FILES_MATCHING REGEX "/[^/]*\\.h$" REGEX "/[^/]*\\.hpp$" REGEX "/[^/]*\\.empbc$")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/emp-tool/ir/files" TYPE FILE FILES "/mnt/c/Functional_Encryption_for_Logistic_Regression/baselines/gc_ag2pc/emp-tool/emp-tool/ir/files/README.md")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/emp-tool/ir/files" TYPE DIRECTORY FILES "/mnt/c/Functional_Encryption_for_Logistic_Regression/baselines/gc_ag2pc/emp-tool/emp-tool/ir/files/manifests" FILES_MATCHING REGEX "/[^/]*\\.json$")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/licenses/emp-tool" TYPE FILE FILES
    "/mnt/c/Functional_Encryption_for_Logistic_Regression/baselines/gc_ag2pc/emp-tool/LICENSE"
    "/mnt/c/Functional_Encryption_for_Logistic_Regression/baselines/gc_ag2pc/emp-tool/THIRD_PARTY_NOTICES.md"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/emp-tool/emp-toolTargets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/emp-tool/emp-toolTargets.cmake"
         "/mnt/c/Functional_Encryption_for_Logistic_Regression/baselines/gc_ag2pc/emp-toolbuild/CMakeFiles/Export/c013aacf683a869882191475c5bbe454/emp-toolTargets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/emp-tool/emp-toolTargets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/emp-tool/emp-toolTargets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/emp-tool" TYPE FILE FILES "/mnt/c/Functional_Encryption_for_Logistic_Regression/baselines/gc_ag2pc/emp-toolbuild/CMakeFiles/Export/c013aacf683a869882191475c5bbe454/emp-toolTargets.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/emp-tool" TYPE FILE FILES "/mnt/c/Functional_Encryption_for_Logistic_Regression/baselines/gc_ag2pc/emp-toolbuild/CMakeFiles/Export/c013aacf683a869882191475c5bbe454/emp-toolTargets-release.cmake")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/emp-tool" TYPE FILE FILES
    "/mnt/c/Functional_Encryption_for_Logistic_Regression/baselines/gc_ag2pc/emp-toolbuild/emp-tool-config.cmake"
    "/mnt/c/Functional_Encryption_for_Logistic_Regression/baselines/gc_ag2pc/emp-toolbuild/emp-tool-config-version.cmake"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/mnt/c/Functional_Encryption_for_Logistic_Regression/baselines/gc_ag2pc/emp-toolbuild/tools/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/mnt/c/Functional_Encryption_for_Logistic_Regression/baselines/gc_ag2pc/emp-toolbuild/test/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/mnt/c/Functional_Encryption_for_Logistic_Regression/baselines/gc_ag2pc/emp-toolbuild/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
