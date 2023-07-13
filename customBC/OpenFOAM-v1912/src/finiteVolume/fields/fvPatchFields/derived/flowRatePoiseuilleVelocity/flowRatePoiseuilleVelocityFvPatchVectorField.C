/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2022 Gustavo Solcia
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

#include "flowRatePoiseuilleVelocityFvPatchVectorField.H"
#include "addToRunTimeSelectionTable.H"
#include "volFields.H"
#include "one.H"

#include "IFstream.H"
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::flowRatePoiseuilleVelocityFvPatchVectorField::
flowRatePoiseuilleVelocityFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedValueFvPatchField<vector>(p, iF),
    flowRate_(),
    circularCrossSection_(false)
{}


Foam::flowRatePoiseuilleVelocityFvPatchVectorField::
flowRatePoiseuilleVelocityFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const dictionary& dict
)
:
    fixedValueFvPatchField<vector>(p, iF, dict, false),
    circularCrossSection_(dict.getOrDefault<Switch>("circularCrossSection", false))
{
    fvPatchVectorField::operator=(vectorField("value", dict, p.size()));
    flowRate_ = Function1<scalar>::New("volumetricFlowRate", dict);
}


Foam::flowRatePoiseuilleVelocityFvPatchVectorField::
flowRatePoiseuilleVelocityFvPatchVectorField
(
    const flowRatePoiseuilleVelocityFvPatchVectorField& ptf,
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    fixedValueFvPatchField<vector>(ptf, p, iF, mapper),
    flowRate_(ptf.flowRate_.clone()),
    circularCrossSection_(ptf.circularCrossSection_)
{}


Foam::flowRatePoiseuilleVelocityFvPatchVectorField::
flowRatePoiseuilleVelocityFvPatchVectorField
(
    const flowRatePoiseuilleVelocityFvPatchVectorField& ptf
)
:
    fixedValueFvPatchField<vector>(ptf),
    flowRate_(ptf.flowRate_.clone()),
    circularCrossSection_(ptf.circularCrossSection_)
{}


Foam::flowRatePoiseuilleVelocityFvPatchVectorField::
flowRatePoiseuilleVelocityFvPatchVectorField
(
    const flowRatePoiseuilleVelocityFvPatchVectorField& ptf,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedValueFvPatchField<vector>(ptf, iF),
    flowRate_(ptf.flowRate_.clone()),
    circularCrossSection_(ptf.circularCrossSection_)
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::flowRatePoiseuilleVelocityFvPatchVectorField::updateCoeffs()
{
    if (updated())
    {
        return;
    }

    const scalar t = db().time().timeOutputValue();
    const vectorField n(patch().nf());
    boundBox bb(patch().patch().localPoints(), true);
    vector ctr = 0.5*(bb.max() + bb.min());
    const vectorField& c = patch().Cf();

    if (circularCrossSection_)
    {
            vector R = 0.5*(bb.max() - bb.min());

            scalarField coord = pow(2,0.5)*Foam::mag(c - ctr)/Foam::mag(R);

            const scalar avgU = -flowRate_->value(t)/gSum(patch().magSf());

            this->operator==(2*n*avgU*(1-coord*coord));
    }
    else
    {
            const scalarField patchArea = patch().magSf();
            const scalar area = gSum(patchArea);
            const scalar avgR = pow(area/Foam::constant::mathematical::pi,0.5);
            const scalar flow = -flowRate_->value(t);
            const scalar avgU = flow/(avgR*avgR*Foam::constant::mathematical::pi);
            scalarField U_hat = patchArea;

            vectorField boundaryPoints;

            IFstream dataStream("boundaryPoints");
            dataStream >> boundaryPoints;

            forAll(c, faceI)
            {
                scalar minDistance = avgR; // just a reference value in scale with the geometry
                for(int j=0; j<boundaryPoints.size(); j++)
                {
                        const scalar rx = c[faceI].x()-boundaryPoints[j].x();
                        const scalar ry = c[faceI].y()-boundaryPoints[j].y();
                        const scalar rz = c[faceI].z()-boundaryPoints[j].z();
                        const scalar radius = pow(rx*rx+ry*ry+rz*rz,0.5);
                        if (radius<minDistance){
                                minDistance=radius;
                        }
                }
                scalar x = c[faceI].x()-ctr.x();
                scalar y = c[faceI].y()-ctr.y();
                scalar z = c[faceI].z()-ctr.z();
                scalar magr = pow(x*x+y*y+z*z,0.5);

                scalar r = magr/(minDistance+magr);
                scalar U_trial = 2*avgU*(1-r*r);

                U_hat[faceI] = U_trial;
            }

            this->operator==(n*U_hat*flow/gSum(U_hat*patchArea));
    }

    fixedValueFvPatchVectorField::updateCoeffs();
}


void Foam::flowRatePoiseuilleVelocityFvPatchVectorField::write(Ostream& os) const
{
    fvPatchField<vector>::write(os);
    flowRate_->writeData(os);
    writeEntry("value", os);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
   makePatchTypeField
   (
       fvPatchVectorField,
       flowRatePoiseuilleVelocityFvPatchVectorField
   );
}


// ************************************************************************* //
