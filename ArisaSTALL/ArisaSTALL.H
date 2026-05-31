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

Class
    Foam::solvers::ArisaSTALL

Description
    ArisaSTALL solver based on standard fluid solver

\*---------------------------------------------------------------------------*/

#ifndef ArisaSTALL_H
#define ArisaSTALL_H

#include "fluid.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace solvers
{

/*---------------------------------------------------------------------------*\
                          Class ArisaSTALL Declaration
\*---------------------------------------------------------------------------*/

class ArisaSTALL
:
    public fluid
{

public:

    //- Runtime type information
    TypeName("ArisaSTALL");


    // Constructors

        //- Construct from region mesh
        ArisaSTALL(fvMesh& mesh);

        //- Disallow default bitwise copy construction
        ArisaSTALL(const ArisaSTALL&) = delete;


    //- Destructor
    virtual ~ArisaSTALL();


    // Member Functions

        //- Disallow default bitwise assignment
        void operator=(const ArisaSTALL&) = delete;

        //- Predict the momentum transport (empty for Euler solver)
        virtual void momentumTransportPredictor();

        //- Construct and optionally solve the momentum equation
        virtual void momentumPredictor();

        //- Correct the momentum transport (empty for Euler solver)
        virtual void momentumTransportCorrector();

        //- Predict thermophysical transport (empty for Euler solver)
        virtual void thermophysicalTransportPredictor();

        //- Construct and solve the energy equation,
        //  convert to temperature
        //  and update thermophysical and transport properties
        virtual void thermophysicalPredictor();

        //- Correct the thermophysical transport (empty for Euler solver)
        virtual void thermophysicalTransportCorrector();

        //- Correct pressure
        virtual void correctPressure();

protected:

    //- Blockage factor field (lambda = actualFlowArea / totalArea)
    volScalarField lambda_;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace solvers
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
