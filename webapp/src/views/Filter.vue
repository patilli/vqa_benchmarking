<template>
  <div>
    <div class="bg">
      <b-container class="bv-example-row">
      <b-row class="mb-3"></b-row>
      <b-row>
        <b-col class="mb-12">
          <h3 class="mt-0 mb-1">Filter for specific data samples</h3>
          Select one of your models, a dataset, and a metric before specifying the minimal and maximal score range 
          and retrieve data samples based on your choices.
        </b-col>
      </b-row>
      <b-row class="mb-3"></b-row>
      <b-row>
        <b-col>
          <b-form-select
            name="Model"
            id="model"
            placeholder="Model"
            v-model="model"
            @change="getSamples()"
          >
            <b-form-select-option value="">Select a model</b-form-select-option>
            <b-form-select-option :value="model">{{model}}</b-form-select-option>
          </b-form-select>
        </b-col>
        <b-col>
          <b-form-select
            name="Dataset"
            id="dataset"
            placeholder="Dataset"
            v-model="dataset"
            @change="getSamples()"
          >
            <b-form-select-option v-for="(name, index) in available_datasets" :key="index" :value="name">{{ name }}</b-form-select-option>
          </b-form-select>
        </b-col>
        <b-col>
          <b-form-select
            name="metric"
            id="metric"
            v-model="metric"
            label="Metric"
            @change="getSamples()"
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
      </b-row>
      <b-row class="mb-3"></b-row>
      <b-row>
        <b-col>
          <label for="range-1">Minimum Score Range {{ min_value }}</label>
          <b-form-input id="range-1" v-model="min_value" type="range" min="0" max="100" step="0.01"></b-form-input>
          <b-form-input id="input-1" type="number" placeholder="" v-model="min_value"></b-form-input>
        </b-col>
        <b-col>
          <label for="range-2">Maximum Score Range {{ max_value }}</label>
          <b-form-input id="range-2" v-model="max_value" type="range" min="0" max="100" step="0.01"></b-form-input>
          <b-form-input id="input-2" type="number" placeholder="Maximum" v-model="max_value"></b-form-input>
        </b-col>
      </b-row>
      <b-row class="mb-3"></b-row>
      <b-row>
        <b-col class="mb-12">
          <b-pagination
              v-model="currentPage"
              :total-rows="n_rows"
              :per-page="perPage"
              aria-controls="data-samples-table"
            ></b-pagination>
        </b-col>
      </b-row>
      <b-row>
        <b-col>
           <b-table id="data-samples-table" :items="pageSamples" :fields="fields" :current-page="currentPage" v-model="currentItems" bordered class="table-striped table-hover table-condensed" @row-clicked="sentView"/>
        </b-col>
      </b-row>
      </b-container>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      filterdata: [],
      model: "",
      metric: "",
      dataset: "",
      min_value: 50.0,
      max_value: 100.0,
      tableData: [],
      perPage: 50,
      currentPage: 1,
      n_rows: 0,
      available_datasets: [],
      currentItems: null,
      fields: [
          { key: 'question_id', label: 'Question ID', sortable: true },
          { key: 'question', label: 'Question', sortable: false },
          { key: 'ground_truth', label: 'Ground Truth', sortable: false },
          { key: 'prediction_class', label: 'Prediction', sortable: false },
          { key: 'score', label: 'Score', sortable: true, formatter: (value, key, item) => value.toFixed(2) }]
    };
  },
  watch: {
    min_value: function(newVal, oldVal) {
      this.getSamples(); 
    },
    max_value: function(newVal, oldVal) {
      this.getSamples();
    }
  },
  computed: {
    pageSamples: function() {
      return this.filterdata.slice((this.currentPage - 1) * this.perPage,
                                    this.currentPage * this.perPage);
    }
  },
  created() {
    this.model = this.$route.params.model;
    this.dataset = this.$route.params.dataset;
    this.metric = this.$route.params.metric;
    this.getInformation();
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
        that.getSamples();
      });
    },
    getSamples() {
      let that = this;
      fetch('http://localhost:44123/filter', {method: 'POST', 
                                                headers: {
                                                'Content-Type': 'application/json'
                                              },
                                              body: JSON.stringify({
                                                'model': this.model,
                                                'dataset': this.dataset,
                                                'metric': this.metric,
                                                'minValue': this.min_value,
                                                'maxValue': this.max_value
                                              }) 
      }).then(res => res.json()).then(function(data) {
        that.filterdata = data.samples;
        that.n_rows = that.filterdata.length
      });
    },
    sentView(sample) {
      let questionIds = this.filterdata.map((s) => s.question_id);
      let questionId = sample.question_id;
      this.$router.push({
        name: "Sample",
        props: true,
        params: {
          questionId: questionId,
          questionIds: questionIds,
          model: this.model,
          dataset: this.dataset
        },
      });
    }
  }
};
</script>
