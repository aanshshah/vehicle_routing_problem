var toggleStat = function(screenname) {

    // Toggle the statistic's row
    // Get the current value
    var current = d3.select("tr[rowstat='" + screenname + "']")
        .attr("chosen")

    // Invert it. When the current toggle status is "true", the comparison
    // below returns "false"; when the current status is "false", it returns
    // "true". A bit opaque, but I can't store proper booleans in HTML attr.
    d3.select("tr[rowstat='" + screenname + "']")
        .attr("chosen", current == "false")

    // Toggle the statistic in the plot
    drawLinePlot(stats, height, width, margin, transDur, reportStats);
};
