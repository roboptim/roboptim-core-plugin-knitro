# Copyright 2014, Thomas Moulard, CNRS JRL UMI3218
#
# This file is part of roboptim-core.
# roboptim-core is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# roboptim-core is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Lesser Public License for more details.
# You should have received a copy of the GNU Lesser General Public License
# along with roboptim-core.  If not, see <http://www.gnu.org/licenses/>.

# Requires at least CMake 2.6 to configure the package.
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

INCLUDE(cmake/base.cmake)
INCLUDE(cmake/boost.cmake)

SET(PROJECT_NAME roboptim-core-plugin-knitro)
SET(PROJECT_DESCRIPTION "RobOptim core KNITRO plug-in")
SET(PROJECT_URL "https://github.com/roboptim/roboptim-core-plugin-knitro")

SET(HEADERS
  )

SET(PKG_CONFIG_ADDITIONAL_VARIABLES plugindir ${PKG_CONFIG_ADDITIONAL_VARIABLES})

SETUP_PROJECT()

# If compiler support symbol visibility, enable it.
INCLUDE(CheckCCompilerFlag)
CHECK_C_COMPILER_FLAG(-fvisibility=hidden HAS_VISIBILITY)
IF(HAS_VISIBILITY)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
ENDIF()

# If compiler support linker flag preventing undefined symbol, go for it.
CHECK_C_COMPILER_FLAG(-Wl,--no-undefined HAS_NO_UNDEFINED)
IF(HAS_NO_UNDEFINED)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--no-undefined")
ENDIF()

# Search for dependencies.
SET(BOOST_COMPONENTS filesystem system unit_test_framework)
SEARCH_FOR_BOOST()

# Search for roboptim-core.
ADD_REQUIRED_DEPENDENCY("roboptim-core >= 3.2")

# Look for KNITRO.
FIND_PATH(KNITRO_DIR include/knitro.h PATHS /usr /usr/local /opt/KNITRO)
MESSAGE(STATUS "KNITRO_DIR: " ${KNITRO_DIR})
IF(NOT KNITRO_DIR)
  MESSAGE(FATAL_ERROR "KNITRO has not been found")
ENDIF()

INCLUDE_DIRECTORIES("${KNITRO_DIR}/include")
LINK_DIRECTORIES("${KNITRO_DIR}/lib/gcc41-sequential")

ADD_SUBDIRECTORY(src)

OPTION(DISABLE_TESTS "Disable test programs" OFF)
IF(NOT DISABLE_TESTS)
  ADD_SUBDIRECTORY(tests)
ELSE()
  MESSAGE(WARNING
    "Tests should only be disabled for speficic cases. Do it at your own risk.")
ENDIF()

SETUP_PROJECT_FINALIZE()
