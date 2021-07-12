<template>
  <div>
  <b-container class='pt-3'>
    <b-table :items="mainData" :fields="fields" v-model="currentItems" bordered class="table-striped table-hover table-condensed" @row-clicked="rowClicked">
      <template v-slot:cell(actions)="{ detailsShowing, item }" >
        <b-btn @click="toggleDetails(item)">{{ detailsShowing ? 'Hide' : 'Show'}} Details</b-btn>
      </template>
      <template v-slot:row-details="{ item }" >
        <b-card>
          <b-table :items="item.detail" :fields="detailFields" bordered/>
        </b-card>
      </template>
    </b-table>
  </b-container>
  </div>
</template>

<script>
// import jsonData from "../../data.json";

export default {
  name: "HelloWorld",
  props: {
    msg: String,
  },
  data() {
    return {
      mainData: [],
      items: [
          // { id: 1, name: 'Povl', age: 26, gender: 'Male', secret: 'I love kittens' },
          // { id: 2, name: 'Charlie', age: 9, gender: 'Female', secret: 'I love cupcakes' },
          // { id: 3, name: 'Max', age: 71, gender: 'Male', secret: 'I love puppies' }
        ],
        currentItems: [],
        fields: [
          { key: 'model.name', label: 'Model', sortable: true },
          // { key: dataset.name', label: 'Dataset', sortable: true },
          { key: 'metrics.accuracy', label: 'Accuracy', sortable: true, formatter: (value, key, item) => value.toFixed(2) },
          { key: 'metrics.biasImage', label: 'Bias Image', sortable: true, formatter: (value, key, item) => value.toFixed(2) },
          { key: 'metrics.biasQuestion', label: 'Bias Question', sortable: true, formatter: (value, key, item) => value.toFixed(2) },
          { key: 'metrics.robustnessNoiseImageImagespace', label: 'Noise Robustness Image (Imagespace)', sortable: true, formatter: (value, key, item) => value.toFixed(2) },
          { key: 'metrics.robustnessNoiseImageFeaturespace', label: 'Noise Robustness Image (Featurespace)', sortable: true, formatter: (value, key, item) => value.toFixed(2)},
          { key: 'metrics.robustnessNoiseText', label: 'Noise Robustness Question', sortable: true, formatter: (value, key, item) => value.toFixed(2)},
          { key: 'metrics.robustnessSears', label: 'Robustness SEARs', sortable: true, formatter: (value, key, item) => value.toFixed(2)},
          { key: 'metrics.uncertainty', label: 'Uncertainty', sortable: true, formatter: (value, key, item) => value.toFixed(2)},
          { key: 'model.parameters', label: 'Parameters', sortable: true, formatter: (value, key, item) => value.toLocaleString() },
          { key: 'actions', label: "Details"}],
        detailFields: [
          { key: 'dataset.name', label: 'Dataset', sortable: true },
          { key: 'metrics.accuracy', label: 'Accuracy', sortable: true, formatter: (value, key, item) => value.toFixed(2) },
          { key: 'metrics.biasImage', label: 'Bias Image', sortable: true, formatter: (value, key, item) => value.toFixed(2) },
          { key: 'metrics.biasQuestion', label: 'Bias Question', sortable: true, formatter: (value, key, item) => value.toFixed(2) },
          { key: 'metrics.robustnessNoiseImageImagespace', label: 'Noise Robustness Image (Imagespace)', sortable: true, formatter: (value, key, item) => value.toFixed(2) },
          { key: 'metrics.robustnessNoiseImageFeaturespace', label: 'Noise Robustness Image (Featurespace)', sortable: true, formatter: (value, key, item) => value.toFixed(2)},
          { key: 'metrics.robustnessNoiseText', label: 'Noise Robustness Question', sortable: true, formatter: (value, key, item) => value.toFixed(2)},
          { key: 'metrics.robustnessSears', label: 'Robustness SEARs', sortable: true, formatter: (value, key, item) => value.toFixed(2)},
          { key: 'metrics.uncertainty', label: 'Uncertainty', sortable: true, formatter: (value, key, item) => value.toFixed(2)}]
    };
  },
  created() {
    let that = this;
    fetch('http://localhost:44123/overview', {method: 'POST', body: JSON.stringify({}) }).then(res => res.json()).then(function(data) {
      that.mainData = data.summary;
      that.mainData.forEach(item => {
        let model = item.model.name;
        item.detail = data.detail[model];
      });
    that.mainData[0].model.parameters = 201723191;
    that.mainData[1].model.parameters = 211166871;
    that.mainData[2].model.parameters = 112167258;
    that.mainData[3].model.parameters = 185847022;
    console.log(that.mainData)
    console.log("maindata")
    });
  },
  methods: {
    toggleDetails(row) {
        if(row._showDetails){
          this.$set(row, '_showDetails', false)
        }else{
          // this.currentItems.forEach(item => {
          //   this.$set(item, '_showDetails', false)
          // })
          this.$nextTick(() => {
            this.$set(row, '_showDetails', true)
          })
        }
      },
      rowClicked(record, index) {
        console.log('row clicked');
        console.log(record, index);
        this.$router.push({
          name: "Metric",
          props: true,
          params: {
            data: JSON.stringify(this.mainData[index])
          }
        })
      }
  },
};
</script>

<style scoped>
tr.collapse.in {
  display:table-row;
}
</style>>