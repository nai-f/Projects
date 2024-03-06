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
    address[] private accountAddresses;

    uint256 private constant MIN_AMOUNT_FOR_FEE = 50;
    uint256 private constant FEE_PERCENTAGE = 2;
    uint256 private constant EXTERNAL_TRANSFER_MIN_AMOUNT = 10e9; // 10 gwei
    uint256 private constant EXTERNAL_TRANSFER_FEE_PERCENTAGE = 10;

    event EtherDeposited(address indexed sender, uint256 amount);
    event EtherWithdrawn(address indexed sender, uint256 amount);
    event EtherSent(address indexed sender, address indexed receiver, uint256 amount);
    event EtherReceived(address indexed sender, address indexed receiver, uint256 amount);

    modifier minAmountForFee(uint256 amount) {
        require(amount >= MIN_AMOUNT_FOR_FEE, "Amount too small for a fee");
        _;
    }

    function calculateFee(uint256 amount, uint256 percentage) private pure returns (uint256) {
        return (amount * percentage) / 100;
    }

    function deposit() public payable {
        if (accounts[msg.sender].balance == 0) {
            accountAddresses.push(msg.sender);
        }
        accounts[msg.sender].balance += msg.value;
        accounts[msg.sender].transactions.push(
            Transaction({receiver: address(0), amount: msg.value, transactionType: "deposit"})
        );
        emit EtherDeposited(msg.sender, msg.value);
    }

    function withdraw(uint256 amount) public {
        require(accounts[msg.sender].balance >= amount, "Insufficient funds");
        accounts[msg.sender].balance -= amount;
        payable(msg.sender).transfer(amount);
        accounts[msg.sender].transactions.push(
            Transaction({receiver: address(0), amount: amount, transactionType: "withdraw"})
        );
        emit EtherWithdrawn(msg.sender, amount);
    }
        //to sned with internally 
    function sendWithFee(address receiver, uint256 amount) public minAmountForFee(amount) {
        uint256 fee = calculateFee(amount, FEE_PERCENTAGE);
        uint256 amountAfterFee = amount - fee;

        require(accounts[msg.sender].balance >= amount, "Insufficient funds");
        accounts[msg.sender].balance -= amount;
        accounts[receiver].balance += amountAfterFee;

        accounts[msg.sender].transactions.push(
            Transaction({receiver: receiver, amount: amount, transactionType: "send"})
        );
        accounts[receiver].transactions.push(
            Transaction({receiver: receiver, amount: amountAfterFee, transactionType: "receive"})
        );
        emit EtherSent(msg.sender, receiver, amount);
        emit EtherReceived(msg.sender, receiver, amountAfterFee);
    }

    function sendExternally(address bankReceiver, address customerReceiver, uint256 amount) public {
        require(amount >= EXTERNAL_TRANSFER_MIN_AMOUNT, "Amount too small for external transfer");

        uint256 fee = calculateFee(amount, EXTERNAL_TRANSFER_FEE_PERCENTAGE);
        uint256 amountAfterFee = amount - fee;

        require(accounts[msg.sender].balance >= amount, "Insufficient funds");
        accounts[msg.sender].balance -= amount;
        BasicBank(bankReceiver).creditExternalTransfer(customerReceiver, amountAfterFee);

        accounts[msg.sender].transactions.push(
            Transaction({receiver: bankReceiver, amount: amount, transactionType: "send externally"})
        );
        emit EtherSent(msg.sender, bankReceiver, amount);
    }

    function creditExternalTransfer(address receiver, uint256 amount) public {
        accounts[receiver].balance += amount;
        accounts[receiver].transactions.push(
            Transaction({receiver: receiver, amount: amount, transactionType: "receive externally"})
        );
        emit EtherReceived(msg.sender, receiver, amount);
    }

    function getAccountInfo(address user) public view returns (uint256, Transaction[] memory) {
        return (accounts[user].balance, accounts[user].transactions);
    }

    function getTotalBankBalance() public view returns (uint256) {
        uint256 totalBalance = 0;
        for (uint256 i = 0; i < accountAddresses.length; i++) {
            totalBalance += accounts[accountAddresses[i]].balance;
        }
        return totalBalance;
    }

    function getTotalDeposits(address user) public view returns (uint256) {
        uint256 totalDeposits = 0;
        for (uint256 i = 0; i < accounts[user].transactions.length; i++) {
            if (keccak256(abi.encodePacked(accounts[user].transactions[i].transactionType)) == keccak256(abi.encodePacked("deposit"))) {
                totalDeposits += accounts[user].transactions[i].amount;
            }
        }
        return totalDeposits;
    }

    function getTotalWithdrawals(address user) public view returns (uint256) {
        uint256 totalWithdrawals = 0;
        for (uint256 i = 0; i < accounts[user].transactions.length; i++) {
            if (keccak256(abi.encodePacked(accounts[user].transactions[i].transactionType)) == keccak256(abi.encodePacked("withdraw"))) {
                totalWithdrawals += accounts[user].transactions[i].amount;
            }
        }
        return totalWithdrawals;
    }
}
