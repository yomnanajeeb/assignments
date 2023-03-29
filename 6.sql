CREATE DATABASE IF NOT EXISTS market;CREATE DATABASE IF NOT EXISTS market;
USE market;

CREATE TABLE IF NOT EXISTS Products (
  ProductID INT PRIMARY KEY,
  ProductName VARCHAR(50) NOT NULL,
  Price DECIMAL(10, 2) DEFAULT 0.00
);

CREATE TABLE IF NOT EXISTS Customers (
  CustomerID INT PRIMARY KEY,
  Customer_Name VARCHAR(50) NOT NULL,
  Email VARCHAR(50) NOT NULL,
  Phone VARCHAR(20) DEFAULT 'N/A'
);

CREATE TABLE IF NOT EXISTS Orders (
  OrderID INT PRIMARY KEY,
  OrderDate DATE NOT NULL,
  CustomerID INT NOT NULL,
  FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID)
);

SELECT a.CustomerID AS customer1, b.CustomerID AS customer2
FROM Customers a
JOIN Customers b ON a.CustomerID = b.CustomerID
WHERE a.CustomerID < b.CustomerID;

SELECT *
FROM Customers a
JOIN Orders b ON a.CustomerID = b.CustomerID;

SELECT a.CustomerID AS customer1, b.CustomerID AS customer2
FROM Customers a
LEFT OUTER JOIN Customers b ON a.CustomerID = b.CustomerID AND a.CustomerID < b.CustomerID;

SELECT *
FROM Customers
NATURAL JOIN Orders;

SELECT *
FROM Customers
CROSS JOIN Orders;