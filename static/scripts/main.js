//Initiate jQuery on load.
$(function() {
  //Translate text with flask route
  $("#train").on("click", function(e) {
    e.preventDefault();
    var fileUrl = document.getElementById("file_url").value;
    var numFields = document.getElementById("numfields").value;
    var numNeigbours = document.getElementById("numNearestNeigbours").value;
    var distanceFunc = document.getElementById("distance").value;
    var trainRequest = { 'url': fileUrl, 'field': numFields, 'neigbour': numNeigbours, 'distance': distanceFunc }
    console.log(trainRequest)
    if (numNeigbours !== "") {
      $.ajax({
        url: '/train',
        method: 'POST',
        headers: {
            'Content-Type':'application/json'
        },
        dataType: 'json',
        data: JSON.stringify(trainRequest),
        success: function(data) { 
          console.log(data)
          // [[1, 'Some distance', 0.78, ...], [], [] ]
          document.getElementById("train-result").textContent = data;
        }

      });
    };
  });

  $("#query").on("click", function(e) {
    e.preventDefault();
    var queryNum =  document.getElementById("query_num").value;
    var queryRequest = { 'query_num':queryNum };
    if (true) {
      $.ajax({
        url: '/queryall',
        method: 'POST',
        headers: {
            'Content-Type':'application/json'
        },
        dataType: 'json',
        data: JSON.stringify(queryRequest),
        success: function(data) { 
          console.log(data)

          var queryTable=document.getElementById('query-table');
          queryTable.innerHTML = '';

          var label_row=document.createElement('tr');
          var labelNode=document.createTextNode('Rid');
          var label_col=document.createElement('td');
          label_col.appendChild(labelNode);
          label_row.appendChild(label_col);
          labelNode=document.createTextNode('Distance');
          label_col=document.createElement('td');
          label_col.appendChild(labelNode);
          label_row.appendChild(label_col);
          labelNode=document.createTextNode('Score');
          label_col=document.createElement('td');
          label_col.appendChild(labelNode);
          label_row.appendChild(label_col);
          labelNode=document.createTextNode('Neighbor');
          label_col=document.createElement('td');
          label_col.appendChild(labelNode);
          label_row.appendChild(label_col);
          labelNode=document.createTextNode('DatasetName');
          label_col=document.createElement('td');
          label_col.appendChild(labelNode);
          label_row.appendChild(label_col);
          labelNode=document.createTextNode('FeatureLen');
          label_col=document.createElement('td');
          label_col.appendChild(labelNode);
          label_row.appendChild(label_col);
          labelNode=document.createTextNode('Timestamp');
          label_col=document.createElement('td');
          label_col.appendChild(labelNode);
          label_row.appendChild(label_col);
          queryTable.appendChild(label_row);


          for(var i=0;i< queryNum ;i++){
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


})