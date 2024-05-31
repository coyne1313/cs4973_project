DROP DATABASE IF EXISTS CONTEXT;
CREATE DATABASE IF NOT EXISTS CONTEXT;

USE CONTEXT;

CREATE TABLE states (
    id INT PRIMARY KEY,
    stateName VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS user (
    email VARCHAR(255) NOT NULL,
    id INT UNIQUE NOT NULL,
    age INT NOT NULL,
    phone int NOT NULL,
    firstName VARCHAR(50) NOT NULL,
    lastName VARCHAR(50) NOT NULL,
    homeStateID int NOT NULL,

    PRIMARY KEY(id),
    FOREIGN KEY (homeStateID) REFERENCES states(id)
);

CREATE TABLE IF NOT EXISTS country (
    id INT UNIQUE NOT NULL,
    name VARCHAR(50) UNIQUE NOT NULL,
    area int,
    population int,
    happinessIndex DECIMAL(5, 1),
    railwayLength int,
    umemlpoymentRate DECIMAL(5, 1),
    bio MULTILINESTRING,
    tips MULTILINESTRING,

    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS countryRanking (
    countryID int NOT NULL,
    rankingNum int,
    id INT UNIQUE NOT NULL,

    PRIMARY KEY(id),
    FOREIGN KEY (countryID) REFERENCES country(id)
);


CREATE TABLE IF NOT EXISTS rankings (
    userID INT NOT NULL,
    rankingID INT UNIQUE NOT NULL,

    PRIMARY KEY(rankingID, userID),
    FOREIGN KEY (userID) REFERENCES user(id),
    FOREIGN KEY (rankingID) REFERENCES countryRanking(id)
);

CREATE TABLE IF NOT EXISTS mover (
    id INT UNIQUE NOT NULL,
    email VARCHAR(50) NOT NULL,
    phone int NOT NULL,
    bio MULTILINESTRING,
    stars int,
    numReviews int,

    PRIMARY KEY(id)
);

CREATE TABLE IF NOT EXISTS moverContact (
    userID int NOT NULL,
    moverID int NOT NULL,
    dateContacted datetime NOT NULL,

    PRIMARY KEY (userID, moverID),
    FOREIGN KEY (userID) REFERENCES user(id),
    FOREIGN KEY (moverID) REFERENCES mover(id)
);

CREATE TABLE IF NOT EXISTS countryAdmin (
    id INT UNIQUE NOT NULL,
    firstName VARCHAR(50),
    lastName VARCHAR(50),
    bio MULTILINESTRING,
    countryID int UNIQUE,

    PRIMARY KEY(id),
    FOREIGN KEY (countryID) REFERENCES country(id)
);

CREATE TABLE IF NOT EXISTS route (
    fromStateID int NOT NULL,
    toCountryID int NOT NULL,
    moverID int NOT NULL,
    cost int,

    PRIMARY KEY (fromStateID, toCountryID, moverID),
    FOREIGN KEY (fromStateID) REFERENCES states(id),
    FOREIGN KEY (toCountryID) REFERENCES country(id),
    FOREIGN KEY (moverID) REFERENCES mover(id)

);

CREATE TABLE IF NOT EXISTS dependant (
    age int NOT NULL,
    dependeeID int NOT NULL,
    id int UNIQUE NOT NULL,

    PRIMARY KEY(id),
    FOREIGN KEY (dependeeID) REFERENCES user(id)
);

CREATE TABLE IF NOT EXISTS language (
    id int UNIQUE NOT NULL,
    name VARCHAR(50) NOT NULL,
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS speaks (
    languageID int NOT NULL,
    countryID int NOT NULL,
    percentage int,

    PRIMARY KEY(languageID, countryID),
    FOREIGN KEY (languageID) REFERENCES language(id),
    FOREIGN KEY (countryID) REFERENCES country(id)
)









