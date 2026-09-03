/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2022-2025 OpenFOAM Foundation
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
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace solvers
{
    defineTypeNameAndDebug(ArisaSTALL, 0);
    addToRunTimeSelectionTable(solver, ArisaSTALL, fvMesh);
}
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::solvers::ArisaSTALL::ArisaSTALL(fvMesh& mesh)
:
    fluid(mesh),
    lambda_
    (
        IOobject
        (
            "lambda",
            mesh.time().constant(),
            mesh,
            IOobject::READ_IF_PRESENT,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedScalar("lambda", dimless, 1.0)
    )
{
    Info<< "Starting ArisaSTALL solver (Euler version with metal blockage)\n" << endl;
    
    if (!lambda_.headerOk())
    {
        IOdictionary dict
        (
            IOobject
            (
                "fvSolution",
                mesh.time().system(),
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE
            )
        );
        scalar lambdaDefault =
            dict.lookupOrDefault<scalar>("lambdaDefault", 1.0);
        Info<< "lambda file not found, using uniform default value: "
            << lambdaDefault << endl;
        lambda_ = lambdaDefault;
    }

    Info<< "Lambda field: min = " << min(lambda_).value()
        << ", max = " << max(lambda_).value() << endl;
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * //

Foam::solvers::ArisaSTALL::~ArisaSTALL()
{}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::solvers::ArisaSTALL::momentumTransportPredictor()
{
    // Empty for Euler solver - no momentum transport model
}


void Foam::solvers::ArisaSTALL::momentumTransportCorrector()
{
    // Empty for Euler solver - no momentum transport model
}


void Foam::solvers::ArisaSTALL::thermophysicalTransportPredictor()
{
    // Empty for Euler solver - no thermophysical transport model
}


void Foam::solvers::ArisaSTALL::thermophysicalTransportCorrector()
{
    // Empty for Euler solver - no thermophysical transport model
}


// ************************************************************************* //
