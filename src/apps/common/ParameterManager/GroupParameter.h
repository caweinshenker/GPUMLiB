/*
	Noel Lopes is a Professor at the Polytechnic of Guarda, Portugal
	and a Researcher at the CISUC - University of Coimbra, Portugal
	Copyright (C) 2009-2015 Noel de Jesus Mendonça Lopes

	This file is part of GPUMLib.

	GPUMLib is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program. If not, see <http://www.gnu.org/licenses/>.
	*/

#ifndef GPUMLIB_GROUP_PARAMETER_H
#define GPUMLIB_GROUP_PARAMETER_H

#include "BaseParameter.h"
#include "qtpropertymanager.h"

#include "assert.h"
#include <vector>

namespace GPUMLib {

	class GroupParameter : public BaseParameter {
	public:
		GroupParameter();

		GroupParameter(const QString & fullname, const QString & name, const QString & summaryDescrition, bool optional);

		virtual ~GroupParameter();

		void AddParameter(BaseParameter * p);

		virtual bool IsValid(LogHTML * log) override;

		virtual void AddToGUIpropertyBrowser(QtTreePropertyBrowser * browser) override;

		virtual void GetParametersValues(QMap<QString, QString> & parametersValues) const override;

	private:
		virtual void InitializeGUIproperty() override {}

		virtual QtAbstractPropertyManager * GetGUIpropertyManager(QtTreePropertyBrowser * browser) override;

		void AddSubproperty(QtTreePropertyBrowser * browser, BaseParameter * subPropParam);

		static QtGroupPropertyManager * propManager;

		std::vector<BaseParameter *> parameters;
	};

} // namespace GPUMLib

#endif // GPUMLIB_GROUP_PARAMETER_H
