-- Setup SQL Script to Create Tables for Rental Property Prediction Project

-- Drop tables if they exist to reset the schema
DROP TABLE IF EXISTS Properties, Amenities, PropertyDetails, Predictions;

-- Table: Properties
CREATE TABLE Properties (
    id SERIAL PRIMARY KEY,
    type VARCHAR(20) NOT NULL,                 -- e.g., BHK1, BHK2, RK1
    locality VARCHAR(100) NOT NULL,            -- neighborhood or area
    activation_date DATE,                      -- listing activation date
    latitude DECIMAL(9, 6),                    -- latitude of property location
    longitude DECIMAL(9, 6),                   -- longitude of property location
    lease_type VARCHAR(20),                    -- e.g., FAMILY, BACHELOR, ANYONE
    rent DECIMAL(10, 2)                        -- rental price (target variable)
);

-- Table: Amenities
CREATE TABLE Amenities (
    id SERIAL PRIMARY KEY,
    property_id INT REFERENCES Properties(id), -- foreign key to Properties
    gym BOOLEAN,                               -- TRUE if gym is available
    lift BOOLEAN,                              -- TRUE if lift is available
    swimming_pool BOOLEAN,                     -- TRUE if swimming pool is available
    negotiable BOOLEAN                         -- TRUE if rent is negotiable
);

-- Table: PropertyDetails
CREATE TABLE PropertyDetails (
    id SERIAL PRIMARY KEY,
    property_id INT REFERENCES Properties(id), -- foreign key to Properties
    furnishing VARCHAR(20),                    -- e.g., fully furnished, unfurnished
    parking BOOLEAN,                           -- TRUE if parking is available
    property_size INT,                         -- size in square feet or meters
    property_age INT,                          -- age of the property in years
    bathroom INT,                              -- number of bathrooms
    facing VARCHAR(20),                        -- facing direction, e.g., north, south
    cup_board BOOLEAN,                         -- TRUE if cupboard is available
    floor INT,                                 -- floor number
    total_floor INT,                           -- total floors in the building
    water_supply VARCHAR(50),                  -- water supply type
    building_type VARCHAR(50),                 -- e.g., Apartment, Individual House
    balconies INT                              -- number of balconies
);

-- Table: Predictions
CREATE TABLE Predictions (
    id SERIAL PRIMARY KEY,
    property_id INT REFERENCES Properties(id), -- foreign key to Properties
    predicted_rent DECIMAL(10, 2),             -- predicted rental price
    prediction_date DATE DEFAULT CURRENT_DATE  -- date of prediction
);

-- Insert any sample data here if needed.
