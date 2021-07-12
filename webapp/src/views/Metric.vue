<template>
  <div>
    <b-navbar toggleable="lg" text-variant="light" variant="light" sticky>
        <b-navbar-brand >Model : {{ model }}</b-navbar-brand>
    </b-navbar>
    <b-container class="bv-example-row">
      <b-row class="mb-3"></b-row>
      <b-row class="mb-3">
        <b-col cols="5">
          <b-form-select
            name="dataset"
            id="dataset"
            v-model="dataset"
            @change="getChart()"
            size="lg"
          >
            <b-form-select-option v-for="(name, index) in available_datasets" :key="index" :value="name">{{ name }}</b-form-select-option>
          </b-form-select>
        </b-col>
        <b-col cols="5">
          <b-form-select
            name="metric"
            id="metric"
            v-model="metric"
            label="Metric"
            @change="getChart()"
            size="lg"
          >
            <b-form-select-option value="accuracy">accuracy</b-form-select-option>
            <b-form-select-option value="image_bias_wordspace">bias image</b-form-select-option>
            <b-form-select-option value="question_bias_imagespace">bias question</b-form-select-option>
            <b-form-select-option value="image_robustness_imagespace">noise robustness image imagespace</b-form-select-option>
            <b-form-select-option value="image_robustness_featurespace">noise robustness image featurespace</b-form-select-option>
            <b-form-select-option value="question_robustness_featurespace">noise robustness text</b-form-select-option>
            <b-form-select-option value="sears">robustness sears</b-form-select-option>
            <b-form-select-option value="uncertainty">uncertainty</b-form-select-option>
          </b-form-select>
        </b-col>
        <b-col>
          <b-button size="lg" variant="outline-dark" @click="filter()">Filter</b-button>
        </b-col>
      </b-row>
      <b-row class="mb-3">
        <b-col>
          <dir id="myRow" class="box"></dir>
        </b-col>
      </b-row>
      <!-- <b-row>
        <b-col>
          <dir id="myDiv" class="box"></dir>
        </b-col>
      </b-row>   -->
    </b-container>
  </div>
</template>

<style scoped>
  .navbar.navbar-light.bg-light{
      background-color: #b3b7bb!important;
  }
  </style>

<script>
import Plotly from "plotly.js";

export default {
  data() {
    return {
      available_datasets: [],
      available_metrics: [],
      mainData: {},
      opentemp: [],
      metric: "",
      dataset: "",
      model: "",
    };
  },
  components: {
    /* eslint-disable */ 
    Plotly,
  },
  created() {
    this.mainData = JSON.parse(this.$route.params.data);
    this.model = this.mainData.model.name
    this.getInformation();
  },
  mounted() {
  },
  methods: {
    getInformation() {
      let that = this;
      fetch('http://localhost:44123/information', {method: 'POST', 
                                                headers: {
                                                'Content-Type': 'application/json'
                                              },
                                              body: JSON.stringify({}) 
      }).then(res => res.json()).then(function(data) {
        that.available_datasets = data.information.datasets;
        that.available_metrics = data.information.metrics;
        that.metric = data.information.metrics[0];
        that.dataset = data.information.datasets[0];
        that.getChart();
      });
    },
    getChart() {
      let that = this;
      console.log(this.model, this.dataset, this.metric);
      fetch('http://localhost:44123/metricsdetail', {method: 'POST', 
                                                     headers: {
                                                      'Content-Type': 'application/json'
                                                    },
                                                    body: JSON.stringify({
                                                      model: this.model,
                                                      dataset: this.dataset,
                                                      metric: this.metric
                                                    }) 
      }).then(res => res.json()).then(function(data) {
        that.opentemp = [data.metric];
        // that.plotGraph();
        that.plotStackedBarGraph();
      });
    },
    plotGraph() {
      Plotly.purge('myDiv');
      const temp = this.opentemp.find( ({ name,dataset }) => name === this.metric 
            && dataset.name === this.dataset );
      if(temp !== undefined){      
            var data = [
              {
                x: temp.plot.x.map(val => val.toFixed(2)),
                y: temp.plot.y.map(val => val.toFixed(2)),
                type: "bar",
              },
            ];
            var layout = {
              title: '',
              xaxis: {
                  type: 'category',
                  title: temp.plot.x_title,
                  showgrid: false,
                  zeroline: true,
                  // range: this.metric == 'sears'? [1,4] : [0, 100]
              },
              yaxis: {
                  title: temp.plot.y_title,
                  showline: false,
                  // range: [0, 100]
              }
            };
            Plotly.newPlot("myDiv", data,layout);
      }
    },
    plotStackedBarGraph() {
      Plotly.purge('myRow');
      console.log(this.opentemp)
      const temp = this.opentemp.find( ({ name,dataset }) => name === this.metric 
            && dataset.name === this.dataset );
      console.log(temp)
      if(temp !== undefined){      
            var data = [
              {
                values: temp.plot.y.map(val => val.toFixed(2)),
                labels: temp.plot.x.map(val => val),
                type: "pie",
                hole: .5,
                automargin: true,
                textinfo: "label+percent",
                textposition: "outside",
              },
            ];
            var layout = {
              margin: {"t": 0, "b": 0, "l": 0, "r": 0},
              showlegend: true
            };
            console.log(data)
            Plotly.newPlot("myRow", data, layout);
      }  
    },
    filter (){
      const model = this.model;
      const dataset = this.dataset;
      const metric = this.metric;
      this.$router.push({
        name: 'Filter',
        props: true,
        params: {
          model,
          dataset,
          metric
        }
      });
    },
  },
};
</script>