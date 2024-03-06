// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract BasicBank {
    struct Transaction {
        address receiver;
        uint256 amount;
        string transactionType;
    }

    struct Account {
        uint256 balance;
        Transaction[] transactions;
    }

    mapping(address => Account) private accounts;
    address[] private accountAddresses; // Array to keep track of all account addresses

    // Modifier to check minimum amount for fee
    modifier minAmountForFee(uint amount) {
        require(amount >= 50, "Amount too small for a fee");
        _;
    }

    // Deposit function
    function deposit() public payable {
        if (accounts[msg.sender].balance == 0) {
            accountAddresses.push(msg.sender);
        }
        accounts[msg.sender].balance += msg.value;
        accounts[msg.sender].transactions.push(
            Transaction({receiver: address(0), amount: msg.value, transactionType: "deposit"})
        );
    }

    // Withdraw function
    function withdraw(uint256 amount) public {
        require(accounts[msg.sender].balance >= amount, "Insufficient funds");
        accounts[msg.sender].balance -= amount;
        payable(msg.sender).transfer(amount);
        accounts[msg.sender].transactions.push(
            Transaction({receiver: address(0), amount: amount, transactionType: "withdraw"})
        );
    }

    // Send function with fee
    function sendWithFee(address receiver, uint amount) public minAmountForFee(amount) {
        uint fee = amount * 2 / 100;
        uint amountAfterFee = amount - fee;

        require(accounts[msg.sender].balance >= amount, "Insufficient funds");
        accounts[msg.sender].balance -= amount;
        accounts[receiver].balance += amountAfterFee;

        accounts[msg.sender].transactions.push(
            Transaction({receiver: receiver, amount: amount, transactionType: "send"})
        );
        accounts[receiver].transactions.push(
            Transaction({receiver: receiver, amount: amountAfterFee, transactionType: "receive"})
        );
    }

    // Function to get account info (for debugging)
    function getAccountInfo(address user) public view returns (uint256, Transaction[] memory) {
        return (accounts[user].balance, accounts[user].transactions);
    }

    // Function to calculate and return the total balance of the bank
    function getTotalBankBalance() public view returns (uint256) {
        uint256 totalBalance = 0;
        for (uint i = 0; i < accountAddresses.length; i++) {
            totalBalance += accounts[accountAddresses[i]].balance;
        }
        return totalBalance;
    }

    // Function to get the total amount of deposits made by a user
    function getTotalDeposits(address user) public view returns (uint256) {
        uint256 totalDeposits = 0;
        for (uint i = 0; i < accounts[user].transactions.length; i++) {
            if (keccak256(abi.encodePacked(accounts[user].transactions[i].transactionType)) == keccak256(abi.encodePacked("deposit"))) {
                totalDeposits += accounts[user].transactions[i].amount;
            }
        }
        return totalDeposits;
    }

    // Function to get the total amount of withdrawals made by a user
    function getTotalWithdrawals(address user) public view returns (uint256) {
        uint256 totalWithdrawals = 0;
        for (uint i = 0; i < accounts[user].transactions.length; i++) {
            if (keccak256(abi.encodePacked(accounts[user].transactions[i].transactionType)) == keccak256(abi.encodePacked("withdraw"))) {
                totalWithdrawals += accounts[user].transactions[i].amount;
            }
        }
        return totalWithdrawals;
    }
}
