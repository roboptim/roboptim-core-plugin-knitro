// Copyright (C) 2016 by Benjamin Chrétien CNRS-AIST JRL.
//
// This file is part of the roboptim.
//
// roboptim is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation either version 3 of the License or
// (at your option) any later version.
//
// roboptim is distributed in the hope that it will be useful
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with roboptim.  If not see <http://www.gnu.org/licenses/>.

#ifndef ROBOPTIM_CORE_PLUGIN_KNITRO_UTIL_HH
#define ROBOPTIM_CORE_PLUGIN_KNITRO_UTIL_HH

#include <string>
#include <map>

#include <boost/assign/list_of.hpp>
#include <boost/preprocessor/stringize.hpp>

#include <knitro.h>

namespace roboptim
{
  /// \brief Return a map from the string representation of KNITRO options to
  /// their actual value (enums).
  /// That way, users can provide the option as a string, rather than needing to
  /// include knitro.h to set the options.
  std::map<std::string, int> knitroParameterMap ()
  {
#define REGISTER_OPTION(OPT) \
    (BOOST_PP_STRINGIZE(OPT), KTR_ ## OPT)

    // Load <algo string algo> map
    // Note: the full list contains over 300 options. We stick to the main ones
    // for now.
    std::map<std::string, int> stringToEnum = boost::assign::map_list_of
      REGISTER_OPTION(ALG_ACTIVE)
      REGISTER_OPTION(ALG_ACT_CG)
      REGISTER_OPTION(ALG_AUTO)
      REGISTER_OPTION(ALG_AUTOMATIC)
      REGISTER_OPTION(ALG_BAR_CG)
      REGISTER_OPTION(ALG_BAR_DIRECT)
      REGISTER_OPTION(ALG_IPCG)
      REGISTER_OPTION(ALG_IPDIRECT)
      REGISTER_OPTION(ALG_MULTI)
      REGISTER_OPTION(GRADOPT_CENTRAL)
      REGISTER_OPTION(GRADOPT_EXACT)
      REGISTER_OPTION(GRADOPT_FORWARD)
      REGISTER_OPTION(HESSIAN_NO_F_ALLOW)
      REGISTER_OPTION(HESSIAN_NO_F_FORBID)
      REGISTER_OPTION(HESSOPT_BFGS)
      REGISTER_OPTION(HESSOPT_EXACT)
      REGISTER_OPTION(HESSOPT_FINITE_DIFF)
      REGISTER_OPTION(HESSOPT_LBFGS)
      REGISTER_OPTION(HESSOPT_PRODUCT)
      REGISTER_OPTION(HESSOPT_SR1)
      REGISTER_OPTION(LINSOLVER_AUTO)
      REGISTER_OPTION(LINSOLVER_DENSEQR)
      REGISTER_OPTION(LINSOLVER_HYBRID)
      REGISTER_OPTION(LINSOLVER_INTERNAL)
      REGISTER_OPTION(LINSOLVER_MA27)
      REGISTER_OPTION(LINSOLVER_MA57);

#undef REGISTER_OPTION

    return stringToEnum;
  }
} // end of namespace roboptim

#endif //! ROBOPTIM_CORE_PLUGIN_KNITRO_UTIL_HH
