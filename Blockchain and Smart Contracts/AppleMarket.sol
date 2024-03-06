// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract AppleMarket {
    struct Order {
        uint256 sellerDeposit;
        uint256 buyerDeposit;
        uint256 apples;
        bool isActive;
        bool sellerConfirmed;
    }

    uint256 public sellerAvailableEther;
    address public seller;
    mapping(address => Order) public orders;

    constructor() payable {
        sellerAvailableEther = msg.value;
        seller = msg.sender;
    }

    function sellerConfirmOrder(uint256 numberApples) public payable {
        require(msg.sender == seller, "Only seller can confirm order");
        uint256 requiredDeposit = numberApples * 0.1 ether;
        require(msg.value == requiredDeposit, "Incorrect deposit amount");
        sellerAvailableEther += msg.value;
        orders[msg.sender].sellerDeposit = requiredDeposit;
        orders[msg.sender].sellerConfirmed = true;
    }

    function placeOrder(uint256 numberApples) public payable {
        require(orders[seller].sellerConfirmed, "Seller must confirm order first");
        require(msg.value >= 2 * numberApples * 0.1 ether, "Insufficient deposit by buyer");
        require(!orders[msg.sender].isActive, "Existing active order for buyer");

        orders[msg.sender] = Order({
            sellerDeposit: orders[seller].sellerDeposit,
            buyerDeposit: msg.value,
            apples: numberApples,
            isActive: true,
            sellerConfirmed: true
        });
    }

    function confirmDelivery() public {
        require(orders[msg.sender].isActive, "No active order for buyer");

        sellerAvailableEther += orders[msg.sender].buyerDeposit;
        payable(seller).transfer(orders[msg.sender].sellerDeposit);
        delete orders[msg.sender];
    }

    function addFunds() public payable {
        require(msg.sender == seller, "Only seller can add funds");
        sellerAvailableEther += msg.value;
    }

    function withdraw(uint256 amount) public {
        require(msg.sender == seller, "Only seller can withdraw");
        require(amount <= sellerAvailableEther, "Insufficient funds to withdraw");

        sellerAvailableEther -= amount;
        (bool success, ) = payable(seller).call{value: amount}("");
        require(success, "Transfer failed");
    }

    function cancelOrder() public {
        require(orders[msg.sender].isActive, "No active order to cancel");
        uint256 refundAmount = orders[msg.sender].buyerDeposit;
        payable(msg.sender).transfer(refundAmount);
        sellerAvailableEther += orders[msg.sender].sellerDeposit;
        delete orders[msg.sender];
    }
}
