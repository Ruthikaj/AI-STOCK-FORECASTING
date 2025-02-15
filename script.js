document.getElementById("stock-form").addEventListener("submit", function (event) {
    event.preventDefault(); // Prevent page reload
    displayTableData();
});

function displayTableData() {
    // Static data for the table
    const stockData = [
        { day: "Day 1", price: 150 },
        { day: "Day 2", price: 155 },
        { day: "Day 3", price: 160 },
        { day: "Day 4", price: 158 },
        { day: "Day 5", price: 162 },
    ];

    // Get table body
    const tableBody = document.querySelector("#stock-table tbody");

    // Clear any previous data
    tableBody.innerHTML = "";

    // Insert rows dynamically
    stockData.forEach((item) => {
        const row = document.createElement("tr");
        row.innerHTML = `
            <td>${item.day}</td>
            <td>${item.price}</td>
        `;
        tableBody.appendChild(row);
    });
}
