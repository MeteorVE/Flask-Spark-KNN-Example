//Initiate jQuery on load.
$(function() {
  //Translate text with flask route
  $("#train").on("click", function(e) {
    e.preventDefault();
    var useAlgo = document.getElementById("algo").getAttribute("name");
    if(useAlgo == 'KNN'){
      var fileUrl = document.getElementById("dataset").value;
      var numNeigbours = document.getElementById("numNearestNeigbours").value;
      var distanceFunc = document.getElementById("distance").value;
      var seed = document.getElementById("seed").value;
      var trainRequest = { 'dataset': fileUrl, 'num_nearest_neigbours': parseInt(numNeigbours), 'distance_func': distanceFunc, 'algorithm': useAlgo, 'seed': parseInt(seed)}
    }
    else if (useAlgo == 'NB'){
      var dataset = document.getElementById("dataset").value;
      var seed = document.getElementById("seed").value;
      var trainRequest = { 'algorithm': useAlgo, 'dataset': dataset, 'seed': parseInt(seed) }
    }
    else if (useAlgo == 'LR') {
      var dataset = document.getElementById("dataset").value;
      var seed = document.getElementById("seed").value;
      var iterations = document.getElementById("iterations").value;
      var trainRequest = { 'algorithm': useAlgo, 'dataset': dataset, 'seed': parseInt(seed), 'iterations': iterations }
    }
    else if (useAlgo == 'DT') {
      var dataset = document.getElementById("dataset").value;
      var seed = document.getElementById("seed").value;
      var categoricalFeaturesInfo = document.getElementById("categoricalFeaturesInfo").value;
      var trainRequest = { 'algorithm': useAlgo, 'dataset': dataset, 'categoricalFeaturesInfo': JSON.parse(categoricalFeaturesInfo),'seed': parseInt(seed)}
    }
    else if (useAlgo == 'RF') {
      var dataset = document.getElementById("dataset").value;
      var seed = document.getElementById("seed").value;
      var categoricalFeaturesInfo = document.getElementById("categoricalFeaturesInfo").value;
      var num_tree = document.getElementById("num_tree").value;
      var trainRequest = { 'algorithm': useAlgo, 'dataset': dataset, 'seed': parseInt(seed), 'numTrees': parseInt(num_tree), 'categoricalFeaturesInfo': JSON.parse(categoricalFeaturesInfo) }
    }
    console.log("trainRequest=",trainRequest)
    if (useAlgo !== "") {
      $.ajax({
        url: '/train',
        method: 'POST',
        headers: {
            'Content-Type':'application/json'
        },
        dataType: 'json',
        data: JSON.stringify(trainRequest),
        success: function(data) { 
          console.log("train response=",data)
          // [[1, 'Some distance', 0.78, ...], [], [] ]
          document.getElementById("train-result").textContent = data;
        }

      });
    };
  });

  $("#query").on("click", function(e) {
    e.preventDefault();
    var queryNum = document.getElementById("query_num").value;
    var queryTableKey = document.getElementById("querySelect").value;
    var queryTableValue = document.getElementById("query_table_key").value;
    var useAlgo = document.getElementById("algo").getAttribute("name");
    var queryRequest = { 'query_num': queryNum, 'query_table_key': queryTableKey, 'query_table_value': queryTableValue, 'use_algo': useAlgo };
    console.log("queryRequest=", queryRequest)

    if (true) {
      $.ajax({
        url: '/query',
        method: 'POST',
        headers: {
            'Content-Type':'application/json'
        },
        dataType: 'json',
        data: JSON.stringify(queryRequest),
        success: function (_data) { 
          console.log("response=", _data)
          data = _data['data']

          var queryTable = document.getElementById('query-table');
          var tr_el = document.getElementById('cloumn-label');
          queryTable.innerHTML = '';
          queryTable.appendChild(tr_el);

          for (var i = 0; i < Object.keys(data).length ;i++){
            var table_row=document.createElement('tr');
            for(var j = 0 ;j < data[i].length;j++){
              var textNode=document.createTextNode(data[i][j]);
              var table_col=document.createElement('td');
              table_col.appendChild(textNode);
              table_row.appendChild(table_col);
            }
            queryTable.appendChild(table_row);
          }

        }

      });
    };
  });

  $("#predict").on("click", function (e) {
    e.preventDefault();
    var useAlgo = document.getElementById("algo").getAttribute("name");
    var feature = document.getElementById("feature").value;
    var predictRequest = {'feature': JSON.parse(feature), 'algorithm': useAlgo}

    console.log("predictRequest=", predictRequest)
    if (useAlgo !== "") {
      $.ajax({
        url: '/predict',
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        dataType: 'json',
        data: JSON.stringify(predictRequest),
        success: function (data) {
          console.log("predict response=", data)
          // label
          document.getElementById("train-result").textContent = data;
        }

      });
    };
  });

})

function querySelectToggle(el) {
  if (el.options[el.selectedIndex].text != 'All') {
    document.getElementById("query_table_key").disabled = false;
  }
  else {
    document.getElementById("query_table_key").setAttribute('disabled', 'disabled');
  }
}