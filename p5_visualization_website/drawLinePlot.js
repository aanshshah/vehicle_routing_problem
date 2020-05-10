// This function builds a line plot on the g.#lineplot built by setupPlot.js
// It creates a line for each row selected in the associated MultiTable.

var drawLinePlot = function(stats, height, width, margin, transDur, reportStats) {

    // Make a list of the selected statistics
    var statsToPlot = [];

    d3.selectAll("tr[chosen='true']")
        .each(function(d) { statsToPlot.push(d.key); });

    // If the function is run without any stats chosen, remove any
    // existing lines and their g elements
    if(statsToPlot.length < 1) { 
        d3.selectAll("path.line")
            .transition().duration(transDur)
                .style("opacity", 1e-6)
                .remove();

        d3.selectAll("g.line").transition().duration(transDur).remove();
               
        return;
    };

    // Subset to the chosen data
    var plotdata = stats.filter(function(d) { 
        return (statsToPlot.indexOf(d.screenname) > -1)
    });


    // Nest each statistic's data in its own object
    var nested = d3.nest()
        .key(function(d) { return d.screenname; })
        .entries(plotdata);
 
    // Set up the x-scale
    var xScale = d3.scale.linear()
        .domain([0, 
                d3.nest()
                .key(function(d) { return d.instance; })
                .entries(plotdata)
                .length])
        .range([0, width - margin.left - margin.right])


    // Set up the y-scale
    var yScale = d3.scale.linear()
        .domain([0, d3.max(plotdata, function(d) { return d.yPlot; }) + d3.deviation(plotdata, function(d) { return d.yPlot; })])
        .range([height - margin.top - margin.bottom, 0])

    // Set up the x-axis
    var xAxis = d3.svg.axis()
        .scale(xScale)
        .orient("bottom")
        .tickPadding(8)
				.tickFormat(function(d) {return d+1;});  // d3.format("d")


				
    // Set up the y-axis
    var yAxis = d3.svg.axis()
        .scale(yScale)
        .orient("right")
        .tickPadding(8);
				
    // Set up a scale function for coloring the paths
    var colorScale = d3.scale.category10();

    var colorStat = {};
    
    for(i = 0; i < reportStats.length; i++) {
        colorStat[reportStats[i]] = colorScale(i);
    }
    
    // Create a line generator
    var generateLine = d3.svg.line()
        .interpolate("step-after")
        .x(function(d, i) { return xScale(i); })
        .y(function(d) { return yScale(d.yPlot); })

    // Create a second line generator for transitioning lines - 
    // this gives them a starting and ending place on the axis
    var generateNullLine = d3.svg.line()
        .interpolate("step-after")
        .x(function(d, i) { return xScale(i); })
        .y(function() { return yScale(0); })
    
    // Find the g.#lineplot element - build the plot on this
    var lineplot = d3.select("#lineplot");

    // Set up a transition function to keep things DRY
    var transit = lineplot.transition().duration(transDur);

    // Each statistic should have a group for its plot elements
    // function(d)... lets d3 key the groups based on which statistic
    // they contain, so that the correct statistics and entered and exited
    // (as opposed to unkeyed joins, which only use the data's array index
    var lineGroups = lineplot.selectAll("g.line")
        .data(nested, function(d) { return d.key; });

    // Add the paths
    lineGroups.enter()
        .append("g")
            .attr("class", "line")
          .append("svg:path")
              .style("opacity", 1e-6)
              .attr("class", "line")
              .attr("d", function(d) { return generateNullLine(d.values); })
              .style("stroke", function(d) { return colorStat[d.key]; })
          .transition().duration(transDur)
              .style("opacity", 1)
              .attr("d", function(d) { return generateLine(d.values); })

//              .style("stroke", function(d) { return colorStat[d.key]; })


    // Transition deselected paths out
    lineGroups.exit().selectAll("path")
        .transition().duration(transDur)
            .attr("d", function(d) { return generateNullLine(d.values); })
        .remove();

    // Remove deselected g elements, too
    lineGroups.exit().transition().duration(transDur).remove();


    // Transition the remaining paths
    transit.selectAll("path.line")
        .attr("d", function(d) { return generateLine(d.values); });

    // Add or transition the x-axis
    if(!xAxisGroup) {
        xAxisGroup = lineplot.append("g")
            .attr("class", "x axis")
            .attr("transform", "translate(0, " + yScale.range()[0] + ")")
            .call(xAxis)
    }
    else {
        transit.select("g.x.axis").call(xAxis)
    }

    // Add or transition the y-axis
    if(!yAxisGroup) {
        yAxisGroup = lineplot.append("g")
            .attr("class", "y axis")
            .attr("transform", "translate(" + xScale.range()[1] + ", 0)")
            .call(yAxis)
    }
    else {
        transit.select("g.y.axis").call(yAxis)
    }

};

