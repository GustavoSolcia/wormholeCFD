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

Class
    Foam::flowRatePoiseuilleVelocityFvPatchVectorField

Group
    grpInletBoundaryConditions

Description
    Velocity inlet boundary condition creating a parabolic velocity field normal to the patch adjusted to match the specified flow rate (m3/s)

Usage
    Example of the boundary condition specification for a volumetric flow rate:
    \verbatim
    <patchName>
    {
        type                flowRatePoiseuilleVelocity;
        volumetricFlowRate  0.2;
        value               uniform (0 0 0);
    }
    \endverbatim

    The \c flowRate entry is a \c Function1 of time, see Foam::Function1Types.

Note
    - The value is positive into the domain (as an inlet)

    See also
    Foam::fixedValueFvPatchField
    Foam::Function1Types
    Foam::flowRateOutletVelocityFvPatchVectorField

SourceFiles
    flowRatePoiseuilleVelocityFvPatchVectorField.C

\*---------------------------------------------------------------------------*/

#ifndef flowRatePoiseuilleVelocityFvPatchVectorField_H
#define flowRatePoiseuilleVelocityFvPatchVectorField_H

#include "fixedValueFvPatchFields.H"
#include "Function1.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
           Class flowRatePoiseuilleVelocityFvPatchVectorField Declaration
\*---------------------------------------------------------------------------*/

class flowRatePoiseuilleVelocityFvPatchVectorField
:
    public fixedValueFvPatchVectorField
{
    // Private data

        //- Inlet integral flow rate
        autoPtr<Function1<scalar>> flowRate_;

        //- Is a circular cross section?
        bool circularCrossSection_;


    // Private member functions


public:

   //- Runtime type information
   TypeName("flowRatePoiseuilleVelocity");


   // Constructors

        //- Construct from patch and internal field
        flowRatePoiseuilleVelocityFvPatchVectorField
        (
            const fvPatch&,
            const DimensionedField<vector, volMesh>&
        );

        //- Construct from patch, internal field and dictionary
        flowRatePoiseuilleVelocityFvPatchVectorField
        (
            const fvPatch&,
            const DimensionedField<vector, volMesh>&,
            const dictionary&
        );

        //- Construct by mapping given
        //  flowRatePoiseuilleVelocityFvPatchVectorField
        //  onto a new patch
        flowRatePoiseuilleVelocityFvPatchVectorField
        (
            const flowRatePoiseuilleVelocityFvPatchVectorField&,
            const fvPatch&,
            const DimensionedField<vector, volMesh>&,
            const fvPatchFieldMapper&
        );

        //- Construct as copy
        flowRatePoiseuilleVelocityFvPatchVectorField
        (
            const flowRatePoiseuilleVelocityFvPatchVectorField&
        );

        //- Construct and return a clone
        virtual tmp<fvPatchVectorField> clone() const
        {
            return tmp<fvPatchVectorField>
            (
                new flowRatePoiseuilleVelocityFvPatchVectorField(*this)
            );
        }

        //- Construct as copy setting internal field reference
        flowRatePoiseuilleVelocityFvPatchVectorField
        (
            const flowRatePoiseuilleVelocityFvPatchVectorField&,
            const DimensionedField<vector, volMesh>&
        );

        //- Construct and return a clone setting internal field reference
        virtual tmp<fvPatchVectorField> clone
        (
            const DimensionedField<vector, volMesh>& iF
        ) const
        {
            return tmp<fvPatchVectorField>
            (
                new flowRatePoiseuilleVelocityFvPatchVectorField(*this, iF)
            );
        }


    // Member functions

        //- Update the coefficients associated with the patch field
        virtual void updateCoeffs();

        //- Write
        virtual void write(Ostream&) const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
