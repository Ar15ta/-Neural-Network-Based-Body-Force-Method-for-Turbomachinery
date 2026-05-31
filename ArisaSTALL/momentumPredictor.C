/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2022-2023 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "ArisaSTALL.H"
#include "fvmDiv.H"
#include "fvcGrad.H"

// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::solvers::ArisaSTALL::momentumPredictor()
{
    volVectorField& U(U_);

    const volScalarField& lambda = lambda_;
    const surfaceScalarField lambdaf(fvc::interpolate(lambda));

    tUEqn =
    (
        lambda*fvm::ddt(rho, U) + fvm::div(lambdaf*phi, U)
      + lambda*MRF.DDt(rho, U)
     ==
        lambda*fvModels().source(rho, U)
    );
    fvVectorMatrix& UEqn = tUEqn.ref();

    UEqn.relax();

    fvConstraints().constrain(UEqn);

    if (pimple.momentumPredictor())
    {
        if (buoyancy.valid())
        {
            solve
            (
                UEqn
             ==
                -fvc::grad(lambda*p) + lambda*rho*buoyancy->g
            );
        }
        else
        {
            solve(UEqn == -fvc::grad(lambda*p));
        }

        fvConstraints().constrain(U);
        K = 0.5*magSqr(U);
    }
}


// ************************************************************************* //
