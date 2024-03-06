import { ethers } from "hardhat";
import { expect } from "chai";
import { loadFixture } from "@nomicfoundation/hardhat-toolbox/network-helpers";


describe ("Ticket", function (){
    async function initializeTicketSale() {
        const [alice, bob, charlie] = await ethers.getSigners ();
        const ticketContract = await ethers.deployContract ("Ticket");
        const ticketSaleContract = await ethers.deployContract ("TicketSale", [ticketContract.getAddress(),]);
        await ticketContract.allow(ticketSaleContract.getAddress(), 5000)

        return { alice, bob, charlie, ticketContract, ticketSaleContract };
    }

    it("bob should get 10 tickets", async function () {
        const { alice, bob, charlie,  ticketContract, ticketSaleContract} = await loadFixture(initializeTicketSale);
        //const number = ethers. formatUnits (10, "gwei");
        await ticketSaleContract.connect(bob).fund(10, { value: 10_000_000_000 });
        const amount = await ticketContract.owners (bob.address);
        expect (amount).to.equal (10);
    });
});
