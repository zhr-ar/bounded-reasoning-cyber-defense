const svg = d3.select("svg"),
    width = +svg.attr("width"),
    height = +svg.attr("height");

const nodes = [
    { id: 1, x: 100, y: 50 },
    { id: 2, x: 300, y: 50 },
    { id: 3, x: 500, y: 50 },
    { id: 4, x: 700, y: 50 },
    { id: 5, x: 900, y: 50 },
    { id: 6, x: 1100, y: 50 },
    { id: 7, x: 600, y: 200 }
];

let links = [
    { source: 7, target: 1, probability: 0 },
    { source: 7, target: 2, probability: 0 },
    { source: 7, target: 3, probability: 0 },
    { source: 7, target: 4, probability: 0 },
    { source: 7, target: 5, probability: 0 },
    { source: 7, target: 6, probability: 0 }
];

let rewardMatrix = [
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
];

const link = svg.append("g")
    .attr("class", "links")
    .selectAll("line")
    .data(links)
    .enter().append("line")
    .attr("class", "link")
    .attr("x1", d => nodes[d.source - 1].x)
    .attr("y1", d => nodes[d.source - 1].y)
    .attr("x2", d => nodes[d.target - 1].x)
    .attr("y2", d => nodes[d.target - 1].y);

const linkText = svg.append("g")
    .attr("class", "link-texts")
    .selectAll("text")
    .data(links)
    .enter().append("text")
    .attr("class", "probability")
    .attr("x", d => (nodes[d.source - 1].x + nodes[d.target - 1].x) / 2)
    .attr("y", d => (nodes[d.source - 1].y + nodes[d.target - 1].y) / 2)
    .attr("text-anchor", "middle")
    .attr("dy", "-0.5em")
    .text(d => d.probability);

const nodeClickCounts = {};

const node = svg.append("g")
    .attr("class", "nodes")
    .selectAll(".node")
    .data(nodes)
    .enter()
    .append(d => d.id === 7 ? document.createElementNS(d3.namespaces.svg, "rect") : document.createElementNS(d3.namespaces.svg, "circle"))
    .attr("class", "node")
    .attr("r", d => d.id === 7 ? null : 20)
    .attr("cx", d => d.id === 7 ? null : d.x)
    .attr("cy", d => d.id === 7 ? null : d.y)
    .attr("x", d => d.id === 7 ? d.x - 20 : null)
    .attr("y", d => d.id === 7 ? d.y - 20 : null)
    .attr("width", d => d.id === 7 ? 40 : null)
    .attr("height", d => d.id === 7 ? 40 : null)
    .attr("fill", d => d.id === 7 ? "red" : "lightgray")
    .each(d => { nodeClickCounts[d.id] = 0; }) // Initialize click counts
    .on("click", function(event, d) {
        if (d.id !== 7) {
            // Reset all other nodes to light gray
            d3.selectAll(".node").attr("fill", function(n) {
                return n.id === 7 ? "red" : "lightgray";
            });
            for (let id in nodeClickCounts) {
                nodeClickCounts[id] = 0;
            }

            nodeClickCounts[d.id] = (nodeClickCounts[d.id] + 1) % 2; // Toggle between 0 and 1
            const newColor = nodeClickCounts[d.id] === 1 ? "lightblue" : "lightgray";
            d3.select(this).attr("fill", newColor);

            if (newColor === "lightblue") {
                showConfirmationBox(d.id);
            } else {
                hideConfirmationBox();
            }
        }
    });

const timestepDisplay = d3.select("#timestepDisplay");

const text = svg.append("g")
    .attr("class", "texts")
    .selectAll("text")
    .data(nodes)
    .enter().append("text")
    .attr("class", "text")
    .attr("x", d => d.x)
    .attr("y", d => d.id === 7 ? d.y + 50 : d.y - 25)
    .attr("text-anchor", "middle")
    .text(d => d.id === 7 ? "Cloud Data Attacker" : `Node ${d.id}`);

const confirmationBox = d3.select("#confirmationBox");
const confirmationText = d3.select("#confirmationText");
const submitBtn = d3.select("#submitBtn");
const cancelBtn = d3.select("#cancelBtn");

let selectedNodeId = null;

function updateTimestepDisplay(timestep, timeMax) {
    // Ensure timestep starts from 1 and does not exceed timeMax
    if (timestep <= timeMax) {
        timestepDisplay.text(`Round: ${timestep} / ${timeMax}`);
    } else {
        timestepDisplay.text(`Round: ${timeMax} / ${timeMax}`);
    }
}

function showConfirmationBox(nodeId) {
    selectedNodeId = nodeId;
    confirmationText.text(`You have selected Node ${nodeId}. Do you want to submit it?`);
    confirmationBox.style("display", "block");
}

function hideConfirmationBox() {
    confirmationBox.style("display", "none");
    selectedNodeId = null;
}

submitBtn.on("click", function() {
    if (selectedNodeId !== null) {
        // Capture the selectedNodeId in a local variable to ensure it doesn't change
        const currentSelectedNodeId = selectedNodeId;
        // Send the selected node to the backend
        fetch('/ch1_node_selected', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ node_id: selectedNodeId })
        })
        .then(response => response.json())  // Make sure to parse the JSON response
        .then(data => {
            if (data.message && data.message.includes('Game Over')) {
                const avgDataProtectionRatio = data.avg_data_prot_w_ratio;
                const completionCode = data.participant_id;  
                alert('Game Over. Thank you for participating!');
                d3.select("#rewardTable").remove();  // Remove the reward table immediately
                d3.select("#timestepDisplay").remove();  // Remove the timestep display
                d3.select("body").append("h2")
                    .attr("class", "game-over-message")
                    .html(`Game Over! Thank you for participating.<br><br>Your Score: ${100*avgDataProtectionRatio} out of 100.<br><br>Your Completion Code: ${completionCode}`)
                    .style("color", "green");

            } else {
                updateTimestepDisplay(data.time_step+1, data.time_max); // Update time step correctly
                console.log('Success:', data);
                updateLinkProbabilities(data.trans_prob); // Ensure the probabilities update

                // Create alert for player's selection and outcome
                resultMessage = (data.attacker_choice === currentSelectedNodeId ? 
                    `🎉 Great job! You won! Attacker chose the same node.\nYou selected Node ${currentSelectedNodeId}.\nYour Reward: ${data.reward}` : 
                    `😢 Oops! You failed. Attacker chose a different node.\nYou selected Node ${currentSelectedNodeId}.\nAttacker chose node ${data.attacker_choice}.\nYour Reward: ${data.reward}`);
                alert(resultMessage);

                // Fetch and update the reward matrix
                fetch('/get_potential_rewards', {
                    method: 'GET'
                })
                .then(response => response.json())
                .then(rewardsData => {
                    console.log('Updated reward matrix:', rewardsData.rewards_matrix);
                    updateRewardTable(rewardsData.rewards_matrix);
                    // Optionally handle attacker choice and reward display
                    // alert(`Attacker chose node ${data.attacker_choice}. Your Reward: ${data.reward}`);
                })
                .catch((error) => {
                    console.error('Error fetching updated reward matrix:', error);
                });
            }
        })
        .catch((error) => {
            console.error('Error:', error);
        });
    }
    hideConfirmationBox();
});


function updateLinkProbabilities(probabilities) {
    links.forEach((link, index) => {
        link.probability = probabilities[index];
    });

    svg.selectAll(".probability")
        .data(links)
        .text(d => {
            if (d.probability === 0) {
                return '0';  // If the probability is zero, display 0
            } else if (d.probability === 1) {
                return '1';  // If the probability is 1, display 1
            } else {
                return d.probability.toExponential(1);  // Use scientific notation with 2 decimals
            }
        });
}




cancelBtn.on("click", function() {
    if (selectedNodeId !== null) {
        d3.select(`.node:nth-child(${selectedNodeId})`).attr("fill", "lightgray");
        nodeClickCounts[selectedNodeId] = 0;
    }
    hideConfirmationBox();
});



// Function to update the reward table
function updateRewardTable(rewards_matrix) {
    rewardMatrix = rewards_matrix;
    const rewardTable = d3.select("#rewardTable");

    // Clear existing table content
    rewardTable.select("thead").html("");
    rewardTable.select("tbody").html("");

    // Add column headers
    rewardTable.select("thead")
        .append("tr")
       
    rewardTable.select("thead tr")
        .selectAll("th.node-header")
        .data([null].concat(nodes.slice(0, 6)))
        .enter()
        .append("th")
        .attr("class", "node-header")
        .text(d => d ? `If Attacker Chooses Node ${d.id}` : "");

    // Add rows
    const rows = rewardTable.select("tbody")
        .selectAll("tr")
        .data(nodes.slice(0, 6))
        .enter()
        .append("tr");

    // Add row headers and cells
    rows.each(function(node, rowIndex) {
        const row = d3.select(this);
        row.append("th").text(`If You Choose Node ${node.id}`);
        row.selectAll("td")
            .data(nodes.slice(0, 6))
            .enter()
            .append("td")
            .text((colNode, colIndex) => rewardMatrix[rowIndex][colIndex]);
    });
}

// Initial render of the reward table
updateRewardTable(rewardMatrix);

// Fetch initial reward matrix
fetch('/get_potential_rewards', {
    method: 'GET'
})
.then(response => response.json())
.then(data => {
    console.log('Initial reward matrix:', data.rewards_matrix);
    updateRewardTable(data.rewards_matrix);
    updateTimestepDisplay(data.time_step, data.time_max); // Initialize timestep display
})
.catch((error) => {
    console.error('Error fetching initial reward matrix:', error);
});
