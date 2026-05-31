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
#include "constrainHbyA.H"
#include "constrainPressure.H"
#include "adjustPhi.H"
#include "fvcMeshPhi.H"
#include "fvcFlux.H"
#include "fvcDdt.H"
#include "fvcGrad.H"
#include "fvcSnGrad.H"
#include "fvcReconstruct.H"
#include "fvcVolumeIntegrate.H"
#include "fvmDiv.H"
#include "fvmLaplacian.H"

// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::solvers::ArisaSTALL::correctPressure()
{
    volScalarField& rho(rho_);
    volScalarField& p(p_);
    volVectorField& U(U_);
    surfaceScalarField& phi(phi_);

    const volScalarField& lambda = lambda_;
    const surfaceScalarField lambdaf(fvc::interpolate(lambda));

    const volScalarField& psi = thermo.psi();
    rho = thermo.rho();
    rho.relax();

    fvVectorMatrix& UEqn = tUEqn.ref();

    // Thermodynamic density needs to be updated by psi*d(p) after the
    // pressure solution
    const volScalarField psip0(psi*p);

    const surfaceScalarField rhof(fvc::interpolate(rho));

    const volScalarField rAU("rAU", 1.0/UEqn.A());
    const surfaceScalarField rhorAUf("rhorAUf", fvc::interpolate(rho*rAU));
    const surfaceScalarField lambdarhorAUf("lambdarhorAUf", lambdaf*rhorAUf);

    tmp<volScalarField> rAtU
    (
        pimple.consistent()
      ? volScalarField::New("rAtU", 1.0/(1.0/rAU - UEqn.H1()))
      : tmp<volScalarField>(nullptr)
    );

    tmp<surfaceScalarField> rhorAtUf
    (
        pimple.consistent()
      ? surfaceScalarField::New("rhoRAtUf", fvc::interpolate(rho*rAtU()))
      : tmp<surfaceScalarField>(nullptr)
    );

    tmp<surfaceScalarField> lambdarhorAtUf
    (
        pimple.consistent()
      ? surfaceScalarField::New("lambdaRhoRAtUf", lambdaf*rhorAtUf())
      : tmp<surfaceScalarField>(nullptr)
    );

    const volScalarField& rAAtU = pimple.consistent() ? rAtU() : rAU;
    const surfaceScalarField& lambdarhorAAtUf =
        pimple.consistent() ? lambdarhorAtUf() : lambdarhorAUf;

    volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U, p));

    if (pimple.nCorrPiso() <= 1)
    {
        tUEqn.clear();
    }

    surfaceScalarField phiHbyA
    (
        "phiHbyA",
        rhof*fvc::flux(HbyA)
      + rhorAUf*fvc::ddtCorr(rho, U, phi, rhoUf)
    );

    MRF.makeRelative(rhof, phiHbyA);

    bool adjustMass = false;

    if (pimple.transonic())
    {
        const surfaceScalarField phidByPsi
        (
            constrainPhid
            (
                fvc::relative(phiHbyA, rho, U)/rhof,
                p
            )
        );

        const surfaceScalarField phid("phid", lambdaf*fvc::interpolate(psi)*phidByPsi);

        phiHbyA -= fvc::interpolate(lambda*psi*p)*phidByPsi;

        if (pimple.consistent())
        {
            phiHbyA += (lambdarhorAAtUf - lambdarhorAUf)*fvc::snGrad(p)*mesh.magSf();
            HbyA += (rAAtU - rAU)*fvc::grad(p);
        }

        constrainPressure(p, rho, U, phiHbyA, lambdarhorAAtUf, MRF);

        fvc::makeRelative(phiHbyA, rho, U);

        fvScalarMatrix pDDtEqn
        (
            lambda*fvc::ddt(rho) + psi*correction(lambda*fvm::ddt(p))
          + fvc::div(lambdaf*phiHbyA) + fvm::div(phid, p)
         ==
            lambda*fvModels().sourceProxy(rho, p)
        );

        while (pimple.correctNonOrthogonal())
        {
            fvScalarMatrix pEqn(pDDtEqn - fvm::laplacian(lambdarhorAAtUf, p));

            pEqn.relax();

            pEqn.setReference
            (
                pressureReference.refCell(),
                pressureReference.refValue()
            );

            fvConstraints().constrain(pEqn);

            pEqn.solve();

            if (pimple.finalNonOrthogonalIter())
            {
                phi = phiHbyA + pEqn.flux();
            }
        }
    }
    else
    {
        if (pimple.consistent())
        {
            phiHbyA += (lambdarhorAAtUf - lambdarhorAUf)*fvc::snGrad(p)*mesh.magSf();
            HbyA += (rAAtU - rAU)*fvc::grad(p);
        }

        constrainPressure(p, rho, U, phiHbyA, lambdarhorAAtUf, MRF);

        fvc::makeRelative(phiHbyA, rho, U);

        if (mesh.schemes().steady())
        {
            adjustMass = adjustPhi(phiHbyA, U, p);
        }

        fvScalarMatrix pDDtEqn
        (
            lambda*fvc::ddt(rho) + psi*correction(lambda*fvm::ddt(p))
          + fvc::div(lambdaf*phiHbyA)
         ==
            lambda*fvModels().sourceProxy(rho, p)
        );

        while (pimple.correctNonOrthogonal())
        {
            fvScalarMatrix pEqn(pDDtEqn - fvm::laplacian(lambdarhorAAtUf, p));

            pEqn.setReference
            (
                pressureReference.refCell(),
                pressureReference.refValue()
            );

            fvConstraints().constrain(pEqn);

            pEqn.solve();

            if (pimple.finalNonOrthogonalIter())
            {
                phi = phiHbyA + pEqn.flux();
            }
        }
    }

    if (!mesh.schemes().steady())
    {
        const bool constrained = fvConstraints().constrain(p);

        thermo_.correctRho(psi*p - psip0);

        if (constrained)
        {
            rho = thermo.rho();
        }

        // Correct density to satisfy continuity after all other rho updates
        fvScalarMatrix rhoEqn
        (
            lambda*fvm::ddt(rho) + fvc::div(lambdaf*phi)
          ==
            lambda*fvModels().source(rho)
        );

        fvConstraints().constrain(rhoEqn);

        rhoEqn.solve();

        fvConstraints().constrain(rho);
    }

    p.relax();

    // Key correction according to Benneke model
    // U = HbyA - lambda * rAAtU * grad(p)
    U = HbyA - lambda*rAAtU*fvc::grad(p);
    U.correctBoundaryConditions();
    fvConstraints().constrain(U);
    K = 0.5*magSqr(U);

    if (mesh.schemes().steady())
    {
        fvConstraints().constrain(p);
    }

    if (adjustMass && !thermo.incompressible())
    {
        p += (initialMass - fvc::domainIntegrate(thermo.rho()))
            /fvc::domainIntegrate(psi);
        p.correctBoundaryConditions();
    }

    if (mesh.schemes().steady() || pimple.simpleRho() || adjustMass)
    {
        rho = thermo.rho();
    }

    if (mesh.schemes().steady() || pimple.simpleRho())
    {
        rho.relax();
    }

    fvc::correctRhoUf(rhoUf, rho, U, phi, MRF);

    if (thermo.dpdt())
    {
        dpdt = fvc::ddt(p);
    }
}

// ************************************************************************* //
