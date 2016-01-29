// Copyright (C) 2016 by Benjamin Chrétien, CNRS-AIST JRL.
//
// This file is part of the roboptim.
//
// roboptim is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// roboptim is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with roboptim.  If not, see <http://www.gnu.org/licenses/>.

#ifndef ROBOPTIM_CORE_PLUGIN_KNITRO_KNITRO_PARAMETERS_UPDATER_HH
# define ROBOPTIM_CORE_PLUGIN_KNITRO_KNITRO_PARAMETERS_UPDATER_HH

# include <string>
# include <stdexcept>

# include <boost/variant/static_visitor.hpp>

# include <knitro.h>

# include <roboptim/core/function.hh>

namespace roboptim
{
  struct KnitroParametersUpdater : public boost::static_visitor<>
  {
    explicit KnitroParametersUpdater (KTR_context* app, const std::string& key)
      : app (app), key (key)
    {
    }

    void operator() (const Function::value_type& val) const
    {
      KTR_set_double_param_by_name (app, key.c_str (), val);
    }

    void operator() (const int& val) const
    {
      KTR_set_int_param_by_name (app, key.c_str (), val);
    }

    template <typename T>
    void operator() (const T&) const
    {
      throw std::runtime_error ("option type not supported by KNITRO");
    }

  private:
    KTR_context* app;
    const std::string& key;
  };
} // end of namespace roboptim

#endif //! ROBOPTIM_CORE_PLUGIN_KNITRO_KNITRO_PARAMETERS_UPDATER_HH
