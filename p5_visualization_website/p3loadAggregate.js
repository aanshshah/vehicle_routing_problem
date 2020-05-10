// Define dimensions of the plot
var margin = {top: 120, right: 60, bottom: 60, left: 60};
var height = 500;
var width = 960;

// Define the transition duration
var transDur = 500;

// Set up a global variable for the names of the stats reported here
// (in hopes of making it easier to keep line colors consistent
var reportStats = [];

var stats;

// Load in the aggregate
d3.csv("p3aggregate.csv", function(crd) {

    // Format the variables as neeeded
    crd.forEach(function(d) {
        d.project = "PROJECT - III: Supply Chain Management";
        d.instanceID = +d.instanceID;
        d.yPlot = +d.yPlot;
    });

    // Assign the data outside of the function for later use
    stats = crd;

    // Load the initial stats table
    makeMultiTable(stats);

    // Make a vector for all of the stats, so that plot attributes can be
    // kept consistent - probably a better way to do this.
    d3.selectAll("tbody tr")
        .each(function(d) { reportStats.push(d.key); });

   // Setup the line plot
	 // Create an svg element for the plot
   d3.select("#plot").append("svg:svg")
       .attr("width", width)
       .attr("height", height)
     .append("g")
        .attr("transform",
              "translate(" + margin.left + "," + margin.top + ")")
        .attr("id", "lineplot");

    // Create global variables for the axes - no need to populate them just yet
    xAxisGroup = null;
    yAxisGroup = null;

});
